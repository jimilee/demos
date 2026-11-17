"""Check the AK timer patch against the REAL hexagon-mlir wrapper generator.

Needs only the hexagon-mlir Python sources (no SDK, no device): set
AKH_HEXMLIR_SRC or HEXAGON_MLIR_ROOT to a checkout. The C++ extension is
stubbed because wrapper generation is pure Python.
"""
import os
import sys
import types
from pathlib import Path

import pytest
import torch

SRC = os.environ.get("AKH_HEXMLIR_SRC") or os.environ.get("HEXAGON_MLIR_ROOT")
BACKEND = Path(SRC or "/nonexistent", "qcom_hexagon_backend", "backend")


@pytest.fixture(scope="module")
def thl():
    if not (BACKEND / "triton_hexagon_launcher.py").exists():
        pytest.skip("hexagon-mlir sources not available (set AKH_HEXMLIR_SRC)")
    try:  # real hexagon triton build present -> use it as is
        from triton.backends.qcom_hexagon_backend import triton_hexagon_launcher as m
        return m
    except ImportError:
        pass
    import triton._C.libtriton as lt
    lt.qcom_hexagon_backend = types.ModuleType("qcom_hexagon_backend")
    pkg = types.ModuleType("triton.backends.qcom_hexagon_backend")
    pkg.__path__ = [str(BACKEND)]
    sys.modules["triton.backends.qcom_hexagon_backend"] = pkg
    from triton.backends.qcom_hexagon_backend import triton_hexagon_launcher as m
    return m


@pytest.mark.parametrize("grid,opts", [((1, 1, 1), {}), ((4, 1, 1), {"enableThreadedDispatch": True}),
                                       ((4, 1, 1), {"enableThreadedDispatch": False})])
def test_patch_rewrites_benchmark_block(thl, grid, opts):
    from akh import hexagon_timer as ht
    from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
    from triton.backends.qcom_hexagon_backend.utils import profile_triton_inputs

    ht.install_timer_patch()
    x = torch.randn(1, 4096).half()
    profs = profile_triton_inputs([x, torch.randn(4096).half(), torch.empty_like(x)])
    o = HexagonOptions().__dict__.copy()
    o.update(opts)
    cpp = thl.TritonHexagonWrapperGenerator(profs, 37, "k", [], grid, o).generate_cpp_wrapper("k", "/data/local/tmp/r")
    assert "benchmark_time_us(" not in cpp  # stock integer-us loop replaced
    assert cpp.count("{") == cpp.count("}")
    t0, t1, loop = cpp.index("ak_t0 ="), cpp.index("ak_t1 ="), cpp.index("ak_i < 37")
    assert t0 < t1 < loop
    if grid[0] > 1:
        call = "tm.exec(" if opts.get("enableThreadedDispatch") else "tm.exec_serial("
        assert cpp.count(call) == 2  # once cold, once in the timed loop
        assert cpp.index("closures[flat_pid]") < t0  # thread-pool setup outside timed region
    assert '"/data/local/tmp/r/perf.txt"' in cpp
