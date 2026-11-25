"""Image normalisation shared by the PyTorch and ONNX backends (NumPy only, so serving does not need PyTorch)."""
from __future__ import annotations

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_array(faces: list[np.ndarray]) -> np.ndarray:
    """List of 224x224 RGB uint8 crops -> normalised float32 array of shape (N, 3, 224, 224)."""
    arr = (np.stack(faces).astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
