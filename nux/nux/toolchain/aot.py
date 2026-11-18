"""Ahead-of-time compilation of linalg-on-tensors IR (vendor-neutral tooling).

Two routes:
  host-mlir-llvm   lower -> LLVM dialect -> object with the MLIR build's OWN LLVM
                   (ExecutionEngine.dump_to_object_file), link a shared library
                   with the system C compiler, call it through ctypes.
  external llc     lower -> LLVM IR text -> `llc` for any target that llc
                   supports (x86-64, hexagon, ...). The MLIR in torch-mlir is
                   newer than the system LLVM, so the IR is downgraded first:
                   new-style float literals are rewritten to exact hex, and
                   newer-only optimization flags/attributes (gep nuw/nusw,
                   icmp samesign, trunc nuw/nsw, captures(...), initializes(...),
                   nocreateundeforpoison, nofpclass, dead_on_unwind, writable)
                   are DROPPED. Dropping them only removes optimizer
                   assumptions; semantics are unchanged. This is verified by
                   compiling the same downgraded IR for x86-64 and comparing
                   outputs with the framework reference.

The Hexagon object from upstream llc is plain LLVM codegen (scalar + LLVM
auto-vectorization for HVX). It is NOT the Hexagon-MLIR pipeline (no VTCM
tiling, DMA, HexKL, ...), and it cannot run here: executing it needs the
Hexagon SDK runtime (FastRPC skel / run_main_on_hexagon) and a device.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import mlir_host as H

TARGETS = {
    "x86_64": ["-march=x86-64", "-relocation-model=pic"],
    # LLVM 18 knows up to hexagonv73 (v75 needs LLVM >= 19)
    "hexagon-v73": ["-march=hexagon", "-mcpu=hexagonv73", "-mattr=+hvxv73,+hvx-length128b"],
    "hexagon-v75": ["-march=hexagon", "-mcpu=hexagonv75", "-mattr=+hvxv75,+hvx-length128b"],
}


def llc_path():
    return shutil.which(os.environ.get("NUX_LLC", "llc"))


def llc_version():
    p = llc_path()
    if not p:
        return None
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"LLVM version (\d+)\.(\d+)\.(\d+)", out)
    return m.group(0).split()[-1] if m else out.splitlines()[0]


def llc_targets():
    p = llc_path()
    if not p:
        return []
    out = subprocess.run([p, "--version"], capture_output=True, text=True).stdout
    return re.findall(r"^\s+(\S+)\s+- ", out, re.M)


def lower_to_llvm_dialect(mod: H.MLIRModule):
    """Returns (ctx, module) lowered with the host pipeline and a C interface on the entry function."""
    from torch_mlir import ir
    from torch_mlir.passmanager import PassManager

    ctx, m = mod.parse()
    with ctx:
        f = H.HostJIT._func(m, mod.func_name)
        f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        PassManager.parse(H.HOST_PIPELINE).run(m.operation)
    return ctx, m


def to_llvm_ir(mod: H.MLIRModule) -> str:
    from torch_mlir.dialects import llvm as L

    ctx, m = lower_to_llvm_dialect(mod)
    with ctx:
        return str(L.translate_module_to_llvmir(m.operation))


# ---- downgrade newer LLVM IR text for an older llc --------------------------
_TOK = re.compile(r"\b(half|bfloat|float|double)\b|\bf0x([0-9A-Fa-f]{8}|[0-9A-Fa-f]{16})\b"
                  r"|(?<![\w.%@\"#])([-+]?\d+\.\d+e[-+]\d+)\b"
                  r"|(?<![\w.%@\"#])([-+]?(?:inf|qnan|snan|nan))\b")
_SPECIAL = {"inf": 0x7FF0000000000000, "+inf": 0x7FF0000000000000, "-inf": 0xFFF0000000000000,
            "qnan": 0x7FF8000000000000, "+qnan": 0x7FF8000000000000, "-qnan": 0xFFF8000000000000,
            "nan": 0x7FF8000000000000, "+nan": 0x7FF8000000000000, "-nan": 0xFFF8000000000000}
_RULES = [
    (re.compile(r"getelementptr (inbounds )?(?:nuw |nusw )+"), lambda m: "getelementptr " + (m.group(1) or "")),
    (re.compile(r"\b(icmp) samesign\b"), r"\1"),
    (re.compile(r"\btrunc (?:nuw |nsw )+"), "trunc "),
    (re.compile(r"\bcaptures\([^)]*\)\s?"), ""),
    (re.compile(r"\binitializes\([^)]*\)\s?"), ""),
    (re.compile(r"\b(?:nocreateundeforpoison|nofpclass\([^)]*\)|dead_on_unwind|dead_on_return|writable)\s?"), ""),
]


def _hexd(v):
    return "0x%016X" % struct.unpack(">Q", struct.pack(">d", v))[0]


def _half_bits(v: float) -> str:
    """LLVM IR half literal (0xHhhhh): IEEE binary16, round-to-nearest-even."""
    return "0xH%04X" % struct.unpack("<H", struct.pack("<e", v))[0]


def _bfloat_bits(v: float) -> str:
    """LLVM IR bfloat literal (0xRhhhh): top 16 bits of binary32, round-to-nearest-even (NaN kept quiet)."""
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    if v != v:
        return "0xR%04X" % ((b >> 16) | 0x40)
    return "0xR%04X" % (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF)


def _narrow(cur, v):
    return _half_bits(v) if cur == "half" else _bfloat_bits(v)


def _fix_line(line):
    cur = ["float"]

    def rep(m):
        if m.group(1):
            cur[0] = m.group(1)
            return m.group(0)
        if m.group(2):
            h = m.group(2)
            v = struct.unpack(">f" if len(h) == 8 else ">d", bytes.fromhex(h))[0]
            return _hexd(v)
        if m.group(4):
            if m.group(4) not in _SPECIAL:
                raise NotImplementedError(f"special FP literal {m.group(4)}")
            if cur[0] in ("half", "bfloat"):
                return _narrow(cur[0], struct.unpack(">d", _SPECIAL[m.group(4)].to_bytes(8, "big"))[0])
            return "0x%016X" % _SPECIAL[m.group(4)]
        v = float(m.group(3))
        if cur[0] == "float":
            v = struct.unpack("f", struct.pack("f", v))[0]
        elif cur[0] in ("half", "bfloat"):
            return _narrow(cur[0], v)
        return _hexd(v)

    return _TOK.sub(rep, line)


def downgrade_llvm_ir(text: str) -> tuple:
    """Returns (text, counts of rewrites per rule)."""
    counts = {}
    out = []
    for line in text.splitlines():
        for i, (pat, rep) in enumerate(_RULES):
            line, n = pat.subn(rep, line)
            if n:
                counts[pat.pattern[:30]] = counts.get(pat.pattern[:30], 0) + n
        out.append(_fix_line(line))
    return "\n".join(out) + "\n", counts


# IEEE-754 maximum/minimum (NaN-propagating, -0 < +0) as plain IR, for llc targets that cannot
# select llvm.maximum/llvm.minimum (LLVM 18 Hexagon: "Cannot select fmaximum").
_FMAXMIN_TMPL = """
define internal {fty} @nux_f{op}_{sfx}({fty} %a, {fty} %b) alwaysinline {{
  %uno = fcmp uno {fty} %a, %b
  %eq = fcmp oeq {fty} %a, %b
  %c = fcmp {cmp} {fty} %a, %b
  %ia = bitcast {fty} %a to {ity}
  %ib = bitcast {fty} %b to {ity}
  %iz = {zop} {ity} %ia, %ib
  %z = bitcast {ity} %iz to {fty}
  %m = select i1 %c, {fty} %a, {fty} %b
  %e = select i1 %eq, {fty} %z, {fty} %m
  %r = select i1 %uno, {fty} {nan}, {fty} %e
  ret {fty} %r
}}
"""
# suffix -> (IR float type, same-width int type, quiet NaN literal)
_FTYPES = {"f32": ("float", "i32", "0x7FF8000000000000"), "f16": ("half", "i16", "0xH7E00")}
# +0/-0 tie: maximum takes AND of the bit patterns (+0 wins), minimum takes OR (-0 wins)
_FOPS = {"maximum": ("ogt", "and"), "minimum": ("olt", "or")}


def _fmaxmin_helper(op: str, sfx: str) -> str:
    fty, ity, nan = _FTYPES[sfx]
    cmp, zop = _FOPS[op]
    return _FMAXMIN_TMPL.format(fty=fty, ity=ity, nan=nan, op=op, sfx=sfx, cmp=cmp, zop=zop)


def expand_fmaxmin(text: str) -> tuple:
    """Replace llvm.maximum/minimum.{f32,f16} with exact IR helpers. Returns (text, n_calls_replaced)."""
    n_total, helpers = 0, []
    for sfx in _FTYPES:
        for op in _FOPS:
            name = f"@llvm.{op}.{sfx}("
            n = text.count(name) - len(re.findall(rf"declare [^\n]*@llvm\.{op}\.{sfx}\(", text))
            if n <= 0:
                continue
            text = re.sub(rf"^declare [^\n]*@llvm\.{op}\.{sfx}\([^\n]*\n", "", text, flags=re.M)
            text = text.replace(name, f"@nux_f{op}_{sfx}(")
            helpers.append(_fmaxmin_helper(op, sfx))
            n_total += n
    return (text + "".join(helpers), n_total) if n_total else (text, 0)


def llc_compile(ll_path: Path, target: str, out: Path, opt="-O2") -> dict:
    p = llc_path()
    if not p:
        raise FileNotFoundError("llc not found (set NUX_LLC)")
    cmd = [p, opt, "-filetype=obj", *TARGETS[target], str(ll_path), "-o", str(out)]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"llc failed: {r.stderr.strip()[:800]}")
    return {"cmd": " ".join(cmd), "seconds": round(time.perf_counter() - t, 2), "stderr": r.stderr.strip()[:400],
            "sha256": H.sha(out.read_bytes()), "bytes": out.stat().st_size, "file": _file_type(out)}


def _file_type(p):
    try:
        return subprocess.run(["file", "-b", str(p)], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return None


def host_object_via_mlir(mod: H.MLIRModule, out: Path, opt_level=2) -> dict:
    """Object file for the host produced by the MLIR build's own LLVM (no downgrade)."""
    from torch_mlir.execution_engine import ExecutionEngine

    ctx, m = lower_to_llvm_dialect(mod)
    t = time.perf_counter()
    with ctx:
        ee = ExecutionEngine(m, opt_level=opt_level)
        ee.dump_to_object_file(str(out))
    return {"route": "mlir-own-llvm", "seconds": round(time.perf_counter() - t, 2), "sha256": H.sha(out.read_bytes()),
            "bytes": out.stat().st_size, "file": _file_type(out)}


def link_shared(objs, out: Path) -> dict:
    cc = shutil.which(os.environ.get("CC", "cc"))
    cmd = [cc, "-shared", *map(str, objs), "-o", str(out), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"link failed: {r.stderr.strip()[:800]}")
    return {"cmd": " ".join(cmd), "sha256": H.sha(out.read_bytes())}


class AOTLibrary:
    """ctypes binding of `_mlir_ciface_<func>(result*, args*...)` from an AOT shared library."""

    def __init__(self, so_path, func_name, result_type: str):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = getattr(self.lib, f"_mlir_ciface_{func_name}")
        self.fn.restype = None
        m = re.match(r"tensor<((?:\d+x)*)(f32|f16|f64)>", result_type)
        if not m:
            raise NotImplementedError(result_type)
        self.out_shape = tuple(int(d) for d in m.group(1).strip("x").split("x") if d)
        self.out_dtype = {"f32": np.float32, "f16": np.float16, "f64": np.float64}[m.group(2)]

    def __call__(self, *arrays):
        from torch_mlir.runtime import (
            as_ctype,
            get_ranked_memref_descriptor,
            make_nd_memref_descriptor,
            ranked_memref_to_numpy,
        )

        res = make_nd_memref_descriptor(len(self.out_shape), as_ctype(self.out_dtype))()
        keep = [np.ascontiguousarray(a) for a in arrays]
        descs = [get_ranked_memref_descriptor(a) for a in keep]
        self.fn(ctypes.byref(res), *[ctypes.byref(d) for d in descs])
        return ranked_memref_to_numpy([res]).copy()


def result_type(mod: H.MLIRModule) -> str:
    from torch_mlir import ir

    ctx, m = mod.parse()
    with ctx:
        f = H.HostJIT._func(m, mod.func_name)
        return str(ir.FunctionType(ir.TypeAttr(f.attributes["function_type"]).value).results[0])


def code_fingerprint() -> dict:
    """Hash of this module (lowering pipeline use, IR downgrade rules, max/min expansion) and the system tools."""
    import hashlib

    cc = shutil.which(os.environ.get("CC", "cc"))
    cc_v = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[:1] if cc else []
    return {"aot_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], "llc": llc_version(),
            "cc": cc_v[0] if cc_v else None}


def workdir_base() -> Path:
    if os.environ.get("NUX_AOT_WORKDIR"):
        return Path(os.environ["NUX_AOT_WORKDIR"])
    return Path(os.environ.get("NUX_CACHE", Path(__file__).resolve().parents[2] / ".cache")) / "aot"


def cleanup_workdirs(paths) -> int:
    """Remove the AOT work directories that contain the given files (only under workdir_base())."""
    base = workdir_base().resolve()
    n = 0
    for d in {Path(p).resolve().parent for p in paths}:
        if base in d.parents and d.exists():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def workdir(prefix="nux_aot_"):
    base = workdir_base()
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def symbols(obj_path, func_name) -> dict | None:
    """Entry symbols matching func_name and undefined symbols of an object file (llvm-nm), or None."""
    nm = shutil.which("llvm-nm-18") or shutil.which("llvm-nm")
    if not nm:
        return None
    defined = subprocess.run([nm, "--defined-only", str(obj_path)], capture_output=True, text=True).stdout
    undef = subprocess.run([nm, "--undefined-only", str(obj_path)], capture_output=True, text=True).stdout
    return {"entry_symbol": [ln.split()[-1] for ln in defined.splitlines() if func_name in ln][:4],
            "undefined_symbols": sorted({ln.split()[-1] for ln in undef.splitlines() if ln.strip()})}
