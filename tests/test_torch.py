"""PyTorch model, inference and training. Skipped where PyTorch is not installed (the serving environment)."""
import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hairify import CLASSES  # noqa: E402
from hairify.model import Classifier, HairifyModel, to_tensor  # noqa: E402


def test_model_shape_and_tensor_normalisation():
    model = HairifyModel().eval()
    x = to_tensor([np.zeros((224, 224, 3), np.uint8)] * 2)
    assert x.shape == (2, 3, 224, 224)
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor(-0.485 / 0.229))
    with torch.no_grad():
        assert model(x).shape == (2, len(CLASSES))


def test_classifier_returns_probabilities():
    clf = Classifier(device="cpu")
    p = clf.predict_proba([np.random.randint(0, 255, (224, 224, 3), np.uint8) for _ in range(3)])
    assert p.shape == (3, 5) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert clf.predict_proba([]).shape == (0, 5)


def test_training_runs_and_saves_a_loadable_checkpoint(tmp_path):
    from hairify.train import train

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for c in CLASSES:
            d = tmp_path / "data" / split / c
            d.mkdir(parents=True)
            for i in range(3):
                cv2.imwrite(str(d / f"{i}.jpg"), rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    for recipe in ("regularised", "original"):
        out = tmp_path / recipe
        summary = train(tmp_path / "data", out, recipe, device="cpu", max_epochs=1, pretrained=False)
        assert summary["train_images"] == 15 and len(summary["history"]) == 2
        assert json.loads((out / "training_summary.json").read_text())["best"]["epoch"] == 1
        clf = Classifier(out / "face_shape_classifier.pt", device="cpu")
        assert clf.predict_proba([np.zeros((224, 224, 3), np.uint8)]).shape == (1, 5)


def test_export_writes_a_reference_for_the_onnx_check(tmp_path):
    pytest.importorskip("torch.onnx")
    from hairify.export import export

    weights = tmp_path / "w.pt"
    torch.save(HairifyModel().state_dict(), weights)
    info = export(weights, tmp_path / "m.onnx")
    assert (tmp_path / "m.onnx").stat().st_size > 1_000_000
    assert np.load(tmp_path / "m.reference_logits.npy").shape == (8, 5) and info["opset"] == 17
