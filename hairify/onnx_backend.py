"""ONNX Runtime backend: same interface as `hairify.model.Classifier`, without PyTorch."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocess import softmax, to_array


class OnnxClassifier:
    def __init__(self, path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.path = Path(path)

    def predict_proba(self, faces: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        out = [softmax(self.session.run(None, {self.input_name: to_array(faces[i:i + batch_size])})[0])
               for i in range(0, len(faces), batch_size)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 5), np.float32)


def load_classifier(path: str | Path, **kwargs):
    """`.onnx` files are served with ONNX Runtime; `.pt` checkpoints with PyTorch."""
    if str(path).lower().endswith(".onnx"):
        return OnnxClassifier(path, **{k: v for k, v in kwargs.items() if k == "threads"})
    from .model import Classifier

    return Classifier(path, **{k: v for k, v in kwargs.items() if k in ("device", "half")})
