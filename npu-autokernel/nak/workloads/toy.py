"""Tiny test workload (LayerNorm -> Linear -> GELU) for fast tests. Not a performance target."""
import hashlib

import torch

from . import Workload, register


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.ln = torch.nn.LayerNorm(16)
        self.fc = torch.nn.Linear(16, 8)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(8, 16, generator=g) * 0.2)
            self.fc.bias.copy_(torch.randn(8, generator=g) * 0.1)

    def forward(self, x):
        return torch.nn.functional.gelu(self.fc(self.ln(x)))


@register("toy-mlp")
class Toy(Workload):
    func_name = "toy_mlp"

    def __init__(self, **_):
        self._m = _Toy().eval()

    def module(self):
        return self._m

    def example_inputs(self):
        return [torch.zeros(1, 4, 16)]

    def meta(self):
        return {"dtype": "float32", "inputs": [[1, 4, 16]], "outputs": [[1, 4, 8]], "quantized": False}

    def corpus(self):
        out = []
        for i, split in enumerate(["tune", "tune", "holdout", "stress"]):
            g = torch.Generator().manual_seed(100 + i)
            x = torch.randn(1, 4, 16, generator=g) * (1e3 if split == "stress" else 1.0)
            out.append({"id": f"toy_{i}", "split": split, "tensors": [x],
                        "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()})
        return out

    def reference(self, inputs, hidden_states=False):
        with torch.no_grad():
            return self._m(*inputs)
