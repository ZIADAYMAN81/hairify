"""Reproducible evaluation on a folder laid out as <root>/<Class>/<image>.

    python -m hairify.evaluate --weights models/face_shape_classifier.pt --data "FaceShape Dataset/testing_set"

Images are cropped with the same face cropper the app uses. Reports accuracy, weighted F1, macro ROC-AUC with
95% bootstrap intervals, per-class metrics and the confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from . import CLASSES
from .face import FaceCropper, NoFaceError
from .model import Classifier

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_folder(root: Path, cropper: FaceCropper, precropped: bool = False):
    faces, labels, skipped = [], [], []
    for i, cls in enumerate(CLASSES):
        for f in sorted((root / cls).iterdir()):
            if f.suffix.lower() not in IMAGE_SUFFIXES:  # e.g. Windows desktop.ini files
                continue
            name = f"{cls}/{f.name}"  # never record local paths in results
            img = cv2.imread(str(f))
            if img is None:
                skipped.append((name, "unreadable")); continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if precropped:
                faces.append(cv2.resize(rgb, (224, 224))); labels.append(i); continue
            try:
                faces.append(cropper.crop(rgb)[0]); labels.append(i)
            except NoFaceError:
                skipped.append((name, "no face"))
    return faces, np.array(labels), skipped


def bootstrap(y, probs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    acc, auc = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(CLASSES):
            continue
        acc.append(accuracy_score(y[idx], probs[idx].argmax(1)))
        auc.append(roc_auc_score(y[idx], probs[idx], multi_class="ovr"))
    q = lambda v: [round(float(x), 4) for x in np.quantile(v, [0.025, 0.975])]  # noqa: E731
    return {"accuracy_95ci": q(acc), "macro_auc_95ci": q(auc)}


def evaluate(classifier: Classifier, root: Path, precropped: bool = False) -> dict:
    faces, y, skipped = load_folder(root, FaceCropper(), precropped)
    probs = classifier.predict_proba(faces)
    pred = probs.argmax(1)
    return {
        "images_evaluated": int(len(y)), "skipped": len(skipped), "skipped_examples": skipped[:10],
        "accuracy": round(accuracy_score(y, pred), 4),
        "weighted_f1": round(f1_score(y, pred, average="weighted"), 4),
        "macro_auc": round(roc_auc_score(y, probs, multi_class="ovr"), 4),
        **bootstrap(y, probs),
        "per_class": {c: {k: round(v, 3) for k, v in m.items() if k != "support"} | {"support": int(m["support"])}
                      for c, m in classification_report(y, pred, target_names=list(CLASSES), output_dict=True).items()
                      if c in CLASSES},
        "confusion_matrix": {"labels": list(CLASSES), "rows_true_columns_predicted": confusion_matrix(y, pred).tolist()},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="folder with one sub-folder per face shape")
    p.add_argument("--precropped", action="store_true", help="images are already face crops")
    p.add_argument("--out", default=None, help="write the results to this JSON file")
    args = p.parse_args(argv)
    res = evaluate(Classifier(args.weights), Path(args.data), args.precropped)
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return res


if __name__ == "__main__":
    main()
