"""Image validation and face cropping.

Fixes a bug in the original notebook: OpenCV's Haar detector returns faces in no particular order, and the
notebook used the first one, so on images with several detections it could crop a neck or a background patch
(13% of the official test images had several detections). The largest detection is now used.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 8000
MARGIN = 0.10  # the same 10% margin the model was trained with
INPUT_SIZE = (224, 224)


class InvalidImageError(ValueError):
    """The upload is not a readable image of acceptable size."""


class NoFaceError(ValueError):
    """No face was found in the image."""


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


def decode_image(data: bytes) -> np.ndarray:
    """Bytes (JPEG, PNG, WebP or BMP) -> RGB array, with size and format checks."""
    if not data:
        raise InvalidImageError("empty upload")
    if len(data) > MAX_BYTES:
        raise InvalidImageError(f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("not a readable image (use JPEG, PNG, WebP or BMP)")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise InvalidImageError(f"image sides must be between {MIN_SIDE} and {MAX_SIDE} pixels, got {w}x{h}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick_face(boxes: list[Box]) -> Box:
    """The largest detection is the subject's face; smaller ones are background faces or false positives."""
    if not boxes:
        raise NoFaceError("no face found: use a clear, front-facing photo")
    return max(boxes, key=lambda b: b.area)


def crop_box(rgb: np.ndarray, box: Box) -> np.ndarray:
    H, W = rgb.shape[:2]
    px, py = int(box.w * MARGIN), int(box.h * MARGIN)
    c = rgb[max(0, box.y - py):min(H, box.y + box.h + py), max(0, box.x - px):min(W, box.x + box.w + px)]
    return cv2.resize(c, INPUT_SIZE)


class FaceCropper:
    def __init__(self, min_size: int = 60):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.detector.empty():
            raise RuntimeError("the OpenCV Haar cascade could not be loaded")
        self.min_size = min_size

    def detect(self, rgb: np.ndarray) -> list[Box]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(self.min_size, self.min_size))
        return [Box(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found]

    def crop(self, rgb: np.ndarray) -> tuple[np.ndarray, Box, int]:
        """Returns (224x224 face crop, chosen box, number of detections)."""
        boxes = self.detect(rgb)
        box = pick_face(boxes)
        return crop_box(rgb, box), box, len(boxes)
