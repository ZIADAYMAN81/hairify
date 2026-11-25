"""Photo bytes -> face crop -> face shape probabilities -> recommendations, with timing for each stage."""
from __future__ import annotations

import time

from .face import FaceCropper, decode_image
from .model import Classifier
from .recommend import recommend_from_probabilities


class Hairify:
    def __init__(self, classifier: Classifier, cropper: FaceCropper | None = None):
        self.classifier = classifier
        self.cropper = cropper or FaceCropper()

    def analyse(self, image_bytes: bytes, top_n: int = 5) -> dict:
        t0 = time.perf_counter()
        rgb = decode_image(image_bytes)
        t1 = time.perf_counter()
        face, box, n_faces = self.cropper.crop(rgb)
        t2 = time.perf_counter()
        probs = self.classifier.predict_proba([face])[0]
        t3 = time.perf_counter()
        result = recommend_from_probabilities(probs, top_n)
        result["face_box"] = {"x": box.x, "y": box.y, "width": box.w, "height": box.h}
        result["faces_detected"] = n_faces
        result["timing_ms"] = {"decode": round((t1 - t0) * 1000, 1), "face_detection": round((t2 - t1) * 1000, 1),
                               "classification": round((t3 - t2) * 1000, 1), "total": round((t3 - t0) * 1000, 1)}
        return result
