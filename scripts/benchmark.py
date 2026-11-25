"""Latency and throughput of the classifier and of the full request pipeline.

    python scripts/benchmark.py --weights models/face_shape_classifier.pt --image photo.jpg --out results/latency.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from hairify.onnx_backend import load_classifier
from hairify.pipeline import Hairify

try:  # PyTorch is optional: the ONNX serving environment does not have it
    import torch
except ImportError:
    torch = None


def sync(cuda: bool):
    if cuda:
        torch.cuda.synchronize()


def timed(fn, repeats: int, warmup: int = 5, cuda: bool = False) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        sync(cuda)
        t = time.perf_counter()
        fn()
        sync(cuda)
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="a .pt checkpoint or an .onnx model")
    p.add_argument("--image", required=True, help="a photo containing one face")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    image = Path(args.image).read_bytes()
    face = [np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)]
    batch = face * 32
    onnx = args.weights.lower().endswith(".onnx")
    res = {"cpu": platform.processor() or platform.machine(), "results": {}}
    if onnx:
        import onnxruntime

        res["onnxruntime"] = onnxruntime.__version__
        configs = [("onnx_cpu_fp32", {}, False)]
    else:
        res["torch"] = torch.__version__
        configs = [("torch_cpu_fp32", {"device": "cpu"}, False)]
        if torch.cuda.is_available():
            res["gpu"] = torch.cuda.get_device_name(0)
            configs += [("torch_cuda_fp32", {"device": "cuda"}, True), ("torch_cuda_fp16", {"device": "cuda", "half": True}, True)]
    for name, kwargs, cuda in configs:
        clf = load_classifier(args.weights, **kwargs)
        engine = Hairify(clf)
        r = {"model_batch1": timed(lambda: clf.predict_proba(face), 30, cuda=cuda),
             "full_request": timed(lambda: engine.analyse(image), 30, cuda=cuda)}
        b = timed(lambda: clf.predict_proba(batch), 10, cuda=cuda)
        r["model_batch32"] = {**b, "images_per_second": round(32 / (b["median_ms"] / 1000), 1)}
        res["results"][name] = r
        print(name, json.dumps(r))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
