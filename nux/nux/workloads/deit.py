"""DeiT-tiny workload (plan §13): facebook/deit-tiny-patch16-224, non-distilled, pinned.

Pinned at the Hub commit that was current when the workload was registered.
Only `pytorch_model.bin` exists in that repo (no safetensors). It is loaded by
transformers with torch.load(weights_only=True) (tensors only), and loading
must report zero missing / unexpected / mismatched keys: transformers 5.x
renamed ViT parameter keys, and a non-strict load silently leaves the model
randomly initialized.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from . import Workload, register

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("NUX_CACHE", ROOT / ".cache"))

MODEL_ID = "facebook/deit-tiny-patch16-224"
REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
FILES = {  # sha256 of the files at REVISION (pytorch_model.bin from the Hub LFS pointer)
    "pytorch_model.bin": "e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c",
    "config.json": None,  # recorded on first fetch into workloads/deit-tiny.lock.json
    "preprocessor_config.json": None,
}
PROCESSOR_BACKEND = "pil"
ATTN_IMPLEMENTATION = "sdpa"  # framework implementation is part of the workload lock
EXPECTED = dict(num_hidden_layers=12, hidden_size=192, num_attention_heads=3, patch_size=16,
                image_size=224, num_labels=1000, tokens=197, head_dim=64)
LOCK_FILE = ROOT / "workloads" / "deit-tiny.lock.json"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def fetch(offline: bool = False) -> Path:
    """Download the pinned snapshot (classic LFS path) and verify hashes."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=list(FILES),
                                  cache_dir=str(CACHE / "hf"), local_files_only=offline))
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    for name, want in FILES.items():
        got = sha256_file(path / name)
        want = want or lock.get("files", {}).get(name)
        if want and got != want:
            raise RuntimeError(f"{name}: sha256 {got} != pinned {want}")
    return path


class DeiTLogits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).logits


@register("deit-tiny")
class DeiTTiny(Workload):
    func_name = "deit_tiny"

    def __init__(self, offline: bool | None = None):
        self.offline = bool(int(os.environ.get("NUX_OFFLINE", "0"))) if offline is None else offline
        self._model = None
        self._path = None
        self._load_info = None

    # ---- model ----------------------------------------------------------
    def path(self):
        if self._path is None:
            try:
                self._path = fetch(offline=True)
            except Exception:  # noqa: BLE001
                if self.offline:
                    raise
                self._path = fetch(offline=False)
        return self._path

    def hf_model(self):
        if self._model is None:
            from transformers import AutoModelForImageClassification

            m, info = AutoModelForImageClassification.from_pretrained(
                str(self.path()), output_loading_info=True, use_safetensors=False,
                local_files_only=True, attn_implementation=ATTN_IMPLEMENTATION)
            bad = {k: v for k, v in info.items() if v}
            if bad:
                raise RuntimeError(f"pretrained weights not fully loaded: { {k: v[:5] for k, v in bad.items()} }")
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self._check_structure(m)
            self._model, self._load_info = m, {k: len(v) for k, v in info.items()}
        return self._model

    @staticmethod
    def _check_structure(m):
        c = m.config
        got = dict(num_hidden_layers=c.num_hidden_layers, hidden_size=c.hidden_size,
                   num_attention_heads=c.num_attention_heads, patch_size=c.patch_size,
                   image_size=c.image_size, num_labels=c.num_labels,
                   tokens=(c.image_size // c.patch_size) ** 2 + 1,
                   head_dim=c.hidden_size // c.num_attention_heads)
        diff = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
        if diff:
            raise RuntimeError(f"DeiT-tiny structure mismatch (got, expected): {diff}")
        if type(m).__name__ != "ViTForImageClassification":
            raise RuntimeError(f"unexpected class {type(m).__name__} (expected non-distilled ViTForImageClassification)")

    def module(self):
        return DeiTLogits(self.hf_model()).eval()

    def example_inputs(self):
        return [torch.zeros(1, 3, 224, 224, dtype=torch.float32)]

    def meta(self):
        return {"dtype": "float32", "inputs": [[1, 3, 224, 224]], "outputs": [[1, 1000]],
                "layout": "NCHW", "batch": 1, "quantized": False, **EXPECTED}

    # ---- sub-workloads for compile smoke (A2) ---------------------------
    def submodules(self):
        """(name, module, example_inputs): minimal op, one encoder block, full model."""
        m = self.hf_model()
        layer0 = m.vit.layers[0] if hasattr(m.vit, "layers") else m.vit.encoder.layer[0]

        class Block(torch.nn.Module):
            def __init__(s, b):
                super().__init__()
                s.b = b

            def forward(s, h):
                out = s.b(h)
                return out[0] if isinstance(out, tuple) else out

        ln = torch.nn.LayerNorm(192, eps=m.config.layer_norm_eps)
        ln.load_state_dict(layer0.layernorm_before.state_dict() if hasattr(layer0, "layernorm_before")
                           else next(mod for mod in layer0.modules() if isinstance(mod, torch.nn.LayerNorm)).state_dict())
        g = torch.Generator().manual_seed(0)
        h = torch.randn(1, 197, 192, generator=g)
        return [("layernorm_192", ln.eval(), [h]), ("encoder_block_0", Block(layer0).eval(), [h]),
                ("deit_tiny_full", self.module(), self.example_inputs())]

    # ---- preprocessing + corpus (A4) -------------------------------------
    def processor(self):
        from transformers import AutoImageProcessor

        if getattr(self, "_proc", None) is None:  # pinned implementation: PIL backend (no torchvision)
            self._proc = AutoImageProcessor.from_pretrained(str(self.path()), local_files_only=True, backend=PROCESSOR_BACKEND)
        return self._proc

    def preprocess(self, pil_image) -> torch.Tensor:
        return self.processor()(images=pil_image, return_tensors="pt")["pixel_values"].float()

    def corpus(self):
        from .corpus import build_image_corpus

        return build_image_corpus(self)

    def reference(self, inputs, hidden_states=False):
        with torch.no_grad():
            out = self.hf_model()(pixel_values=inputs[0], output_hidden_states=hidden_states)
        return (out.logits, out.hidden_states) if hidden_states else out.logits

    def lock(self):
        import transformers

        p = self.path()
        return {"workload_id": self.workload_id, "model_id": MODEL_ID, "revision": REVISION,
                "files": {n: sha256_file(p / n) for n in FILES},
                "framework": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "class": "ViTForImageClassification", "attn_implementation": ATTN_IMPLEMENTATION,
                              "processor": type(self.processor()).__name__, "processor_backend": PROCESSOR_BACKEND},
                "load_info": self._load_info, "structure": EXPECTED, "meta": self.meta(),
                "preprocessor_config": json.loads((p / "preprocessor_config.json").read_text())}


def write_lock(w: DeiTTiny):
    w.hf_model()
    lock = w.lock()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps(lock, indent=2))
    return lock

