"""MLIR host tooling on the toy model; DeiT workload checks when the pinned weights are cached."""
import numpy as np
import pytest

from nux.toolchain import mlir_host as H
from nux.workloads import get


def test_recipes_preserve_semantics_on_host_jit():
    w = get("toy-mlp")
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    assert H.verify(mod)[0]
    x = w.corpus()[0]["tensors"][0]
    ref = w.reference([x]).numpy()
    for r in H.RECIPES.values():
        if H.recipe_applicable(r, mod)[0]:
            m2 = H.apply_recipe(mod, r)
            assert np.abs(H.HostJIT(m2)(x.numpy()) - ref).max() < 1e-5, r.recipe_id


def test_reshape_reproducers_parse():
    w = get("toy-mlp")
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    reps = H.extract_reshape_reproducers(mod, min_rank=99, include_unit_dims=True)
    assert reps and all(r["parses"] is True for r in reps)


def _deit(workload_id="deit-tiny"):
    try:
        w = get(workload_id, offline=True)
        w.path()
        return w
    except Exception:  # noqa: BLE001
        pytest.skip("pinned DeiT-tiny weights not cached (run: python -m nux workload)")


def test_deit_strict_load_and_structure():
    w = _deit()
    m = w.hf_model()
    assert type(m).__name__ == "ViTForImageClassification"
    lock = w.lock()
    assert lock["load_info"] == {"missing_keys": 0, "unexpected_keys": 0, "mismatched_keys": 0, "error_msgs": 0}
    assert lock["files"]["pytorch_model.bin"] == "e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c"


def test_deit_corpus_deterministic_and_reference_finite():
    w = _deit()
    a, b = w.corpus(), w.corpus()
    assert [x["sha256"] for x in a] == [x["sha256"] for x in b]
    assert {x["split"] for x in a} == {"tune", "holdout", "stress"}
    y = w.reference(a[0]["tensors"])
    assert y.shape == (1, 1000) and bool(y.isfinite().all())


def test_deit_fp16_workload_is_separate_and_does_not_touch_fp32():
    """deit-tiny-fp16: fp16 inputs/module, fp32 reference, and casting never mutates the fp32 model."""
    import torch

    from nux.core import contract

    w16 = _deit("deit-tiny-fp16")
    assert w16.meta()["dtype"] == "float16" and w16.example_inputs()[0].dtype == torch.float16
    subs = w16.submodules()
    assert all(p.dtype == torch.float16 for _, m, _ in subs for p in m.parameters())
    assert all(p.dtype == torch.float32 for p in w16.hf_model().parameters())  # shared fp32 model untouched
    corp = w16.corpus()
    assert all(t.dtype == torch.float16 for c in corp for t in c["tensors"])
    x = corp[0]["tensors"]
    ref = w16.reference(x)
    assert ref.dtype == torch.float32
    with torch.no_grad():
        out = w16.module()(x[0])
    r = contract.compare(contract.get("deit-tiny-fp16-logits-v0"), out, ref)
    assert out.dtype == torch.float16 and r["ok"] and r["top1_match"], r
