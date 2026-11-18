"""AOT routes: host .so, LLVM IR downgrade for the system llc, exact max/min expansion, Hexagon object."""
import ctypes
import itertools
import struct

import numpy as np
import pytest
import torch

from nux import aot
from nux import mlir_host as H
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
    from nux import contract, registry, stats
    from nux.controller import Controller
    from nux.store import Store

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
