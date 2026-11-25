"""The EfficientNet-B3 classifier (same architecture as the trained checkpoint) and batched inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import CLASSES
from .preprocess import to_array



class HairifyModel(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), dropout: float = 0.4, pretrained: bool = False):
        super().__init__()
        import timm

        self.base = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0, global_pool="avg")
        f = self.base.num_features
        self.head = nn.Sequential(nn.Linear(f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout * 0.5),
                                  nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.base(x))


def to_tensor(faces: list[np.ndarray]) -> torch.Tensor:
    """List of 224x224 RGB uint8 crops -> normalised NCHW float tensor."""
    return torch.from_numpy(to_array(faces))


class Classifier:
    """Wraps the network for inference. `half=True` uses float16 on a GPU."""

    def __init__(self, weights: str | Path | None = None, device: str | None = None, half: bool = False,
                 model: nn.Module | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = model if model is not None else HairifyModel()
        if weights is not None:
            net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self.half = half and self.device.type == "cuda"
        net = net.to(self.device).eval()
        if self.half:
            net = net.half()
        self.model = net.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def predict_proba(self, faces: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(faces), batch_size):
            x = to_tensor(faces[i:i + batch_size]).to(self.device).to(memory_format=torch.channels_last)
            if self.half:
                x = x.half()
            out.append(torch.softmax(self.model(x).float(), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)
