"""Export the PyTorch classifier to ONNX (needs PyTorch; the ONNX file is then served without it).

    python -m hairify.export --weights models/face_shape_classifier.pt --out models/face_shape_classifier.onnx

The graph takes a float32 batch of normalised 224x224 RGB images (N, 3, 224, 224) and returns logits (N, 5).
The batch dimension is dynamic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import HairifyModel, to_tensor

OPSET = 17


def export(weights: str | Path, out: str | Path) -> dict:
    model = HairifyModel()
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), opset_version=OPSET, input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True,
                      dynamo=False)  # the TorchScript exporter this model was verified with (newer PyTorch defaults to dynamo)
    # Reference logits for a fixed input, so the ONNX side can verify it reproduces PyTorch exactly.
    rng = np.random.default_rng(0)
    faces = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    with torch.no_grad():
        ref = model(to_tensor(faces)).numpy()
    np.save(out.with_suffix(".reference_logits.npy"), ref)
    info = {"onnx": str(out.name), "opset": OPSET, "size_mb": round(out.stat().st_size / 1e6, 1)}
    print(json.dumps(info))
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    export(args.weights, args.out)


if __name__ == "__main__":
    main()
