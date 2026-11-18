"""AOT routes: host .so, LLVM IR downgrade for the system llc, exact max/min expansion, Hexagon object."""
import ctypes
import itertools
import struct

import numpy as np
import pytest
import torch

from nux.toolchain import aot
from nux.toolchain import mlir_host as H
from nux.workloads import get

needs_llc = pytest.mark.skipif(not aot.llc_path(), reason="llc not installed")


def test_downgrade_rewrites():
    src = ("@a = private constant [2 x float] [float f0x3F800000, float -2.468270e-01]\n"
           "@b = private constant float -inf\n"
           "  %1 = fadd double %0, 1.000000e-01\n"
           "  %2 = getelementptr inbounds nuw float, ptr %p, i64 %i\n"
           "  %3 = icmp samesign ult i64 %x, %y\n"
           "attributes #0 = { nocreateundeforpoison nounwind }\n")
    out, counts = aot.downgrade_llvm_ir(src)
    assert "f0x" not in out and "nuw" not in out and "samesign" not in out and "nocreateundeforpoison" not in out
    assert "float 0x3FF0000000000000" in out  # 1.0f exact
    v = struct.unpack("f", struct.pack("f", -2.468270e-01))[0]
    assert "0x%016X" % struct.unpack(">Q", struct.pack(">d", v))[0] in out  # rounded to float, then exact hex
    assert "double %0, 0x3FB999999999999A" in out  # double literal NOT rounded to float
    assert "float 0xFFF0000000000000" in out


def test_downgrade_half_and_bfloat_literals():
    src = ("  %1 = fadd half %0, 1.000000e+00\n  %2 = select i1 %c, half -inf, half 0xH3C00\n"
           "  @g = constant [3 x half] [half 6.103520e-05, half 6.550400e+04, half -2.468750e-01]\n"
           "  %5 = fadd bfloat %b, 1.500000e+00\n  %6 = fadd float %a, 1.000000e+00\n")
    out, _ = aot.downgrade_llvm_ir(src)
    assert "half %0, 0xH3C00" in out and "half 0xHFC00" in out
    assert "[half 0xH0400, half 0xH7BFF, half 0xHB3E6]" in out  # min normal, max finite, rounded
    assert "bfloat %b, 0xR3FC0" in out and "float %a, 0x3FF0000000000000" in out  # type tracked per token


@needs_llc
def test_fmaxmin_expansion_bit_exact():
    src = ("declare float @llvm.maximum.f32(float, float)\ndeclare float @llvm.minimum.f32(float, float)\n"
           "define float @X_max(float %a, float %b) { %r = call float @llvm.maximum.f32(float %a, float %b) ret float %r }\n"
           "define float @X_min(float %a, float %b) { %r = call float @llvm.minimum.f32(float %a, float %b) ret float %r }\n")
    exp, n = aot.expand_fmaxmin(src.replace("@X_", "@new_"))
    assert n == 2
    d = aot.workdir()
    (d / "ref.ll").write_text(src.replace("@X_", "@ref_"))
    (d / "new.ll").write_text(exp)
    for k in ("ref", "new"):
        aot.llc_compile(d / f"{k}.ll", "x86_64", d / f"{k}.o")
    aot.link_shared([d / "ref.o", d / "new.o"], d / "lib.so")
    L = ctypes.CDLL(str(d / "lib.so"))
    for nm in ("ref_max", "ref_min", "new_max", "new_min"):
        getattr(L, nm).restype = ctypes.c_float
        getattr(L, nm).argtypes = [ctypes.c_float, ctypes.c_float]
    vals = [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 3.5, 1e-45, -1e-45]
    bits = lambda f: struct.unpack("I", struct.pack("f", f))[0]  # noqa: E731
    for a, b in itertools.product(vals, vals):
        for op in ("max", "min"):
            r, q = getattr(L, "ref_" + op)(a, b), getattr(L, "new_" + op)(a, b)
            assert (r != r and q != q) or bits(r) == bits(q), (op, a, b, r, q)



@needs_llc
def test_fmaxmin_expansion_bit_exact_f16():
    """f16 helpers (deit-tiny-fp16): compare with llvm.maximum/minimum.f16 on x86 over special and ordinary values.
    Half values cross the C ABI as i16 bit patterns."""
    body = ("define i16 @X_{op}(i16 %ai, i16 %bi) {{ %a = bitcast i16 %ai to half %b = bitcast i16 %bi to half "
            "%r = call half @llvm.{op}imum.f16(half %a, half %b) %o = bitcast half %r to i16 ret i16 %o }}\n")
    src = ("declare half @llvm.maximum.f16(half, half)\ndeclare half @llvm.minimum.f16(half, half)\n"
           + body.format(op="max") + body.format(op="min"))
    exp, n = aot.expand_fmaxmin(src.replace("@X_", "@new_"))
    assert n == 2 and "@llvm.maximum.f16" not in exp
    d = aot.workdir()
    (d / "ref.ll").write_text(src.replace("@X_", "@ref_"))
    (d / "new.ll").write_text(exp)
    for k in ("ref", "new"):
        aot.llc_compile(d / f"{k}.ll", "x86_64", d / f"{k}.o")
    if "hexagon" in aot.llc_targets():  # the expanded form must be selectable by the Hexagon backend
        aot.llc_compile(d / "new.ll", "hexagon-v73", d / "new_hex.o")
    aot.link_shared([d / "ref.o", d / "new.o"], d / "lib16.so")
    L = ctypes.CDLL(str(d / "lib16.so"))
    for nm in ("ref_max", "ref_min", "new_max", "new_min"):
        getattr(L, nm).restype = ctypes.c_uint16
        getattr(L, nm).argtypes = [ctypes.c_uint16, ctypes.c_uint16]
    vals = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 3.5, 6e-8, -6e-8, 65504.0], np.float16)
    vals = [int(v) for v in vals.view(np.uint16)]
    isnan = lambda b: (b & 0x7C00) == 0x7C00 and (b & 0x3FF) != 0  # noqa: E731
    for a, b in itertools.product(vals, vals):
        for op in ("max", "min"):
            r, q = getattr(L, "ref_" + op)(a, b), getattr(L, "new_" + op)(a, b)
            assert (isnan(r) and isnan(q)) or r == q, (op, hex(a), hex(b), hex(r), hex(q))

def test_host_aot_matches_reference():
    w = get("toy-mlp")
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    d = aot.workdir()
    aot.host_object_via_mlir(mod, d / "m.o")
    aot.link_shared([d / "m.o"], d / "libm.so")
    x = w.corpus()[0]["tensors"][0]
    y = aot.AOTLibrary(d / "libm.so", mod.func_name, aot.result_type(mod))(x.numpy())
    assert np.abs(y - w.reference([x]).numpy()).max() < 1e-5


@needs_llc
def test_hexagon_object_and_twin(tmp_path):
    if "hexagon" not in aot.llc_targets():
        pytest.skip("llc without hexagon target")
    from nux.core import contract, stats
    from nux.engine import registry
    from nux.engine.controller import Controller
    from nux.storage.store import Store

    b = registry.resolve({"backend_id": "llvm-aot-hexagon"})
    c = Controller(b, get("toy-mlp"), contract.get("toy-fp32-v0"), Store(tmp_path), stats.Policy(), {"backend_id": "x"})
    r = c.run_baseline()
    assert r.state == "target_compiled"
    st = {s["stage"]: s["status"] for s in r.stages}
    assert st["twin_correctness"] == "passed" and st["device_correctness"] == "pending"
    obj = c.best_artifact.files["hexagon_object"]
    assert "QUALCOMM DSP6" in (aot._file_type(obj) or "")


def test_torch_mlir_maximum_nan_divergence_is_tracked():
    """Observed front-end behavior (torch-mlir 20261101): torch.maximum lowers to cmpf+select and drops a
    NaN in the FIRST operand; torch propagates it. DeiT does not use torch.maximum (softmax uses
    NaN-propagating arith.maximumf). If this test starts failing, the lowering changed: update the note."""

    class MX(torch.nn.Module):
        def forward(self, a, b):
            return torch.maximum(a, b)

    a = torch.tensor([[float("nan"), 1.0]])
    b = torch.tensor([[2.0, float("nan")]])
    mod = H.export(MX(), [a, b], "mx")
    y = H.HostJIT(mod)(a.numpy(), b.numpy())
    assert np.isnan(MX()(a, b).numpy()).all()  # torch: NaN both
    assert not np.isnan(y[0, 0]) and np.isnan(y[0, 1])  # torch-mlir lowering: NaN dropped when first operand is NaN
