"""ONNX Runtime backend. Skipped where ONNX Runtime is not installed (the training environment).

A tiny ONNX graph with the same input and output contract as the real model (images (N,3,224,224) -> logits
(N,5)) is built here, so the tests need neither PyTorch nor the 46 MB trained model.
"""
import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from hairify.onnx_backend import OnnxClassifier, load_classifier  # noqa: E402


@pytest.fixture
def tiny_model(tmp_path):
    """Global average pool over each colour channel, then a fixed 3 -> 5 linear layer."""
    w = np.array([[1, 0, 0, 0, -1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32)  # (3, 5)
    nodes = [helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
             helper.make_node("Flatten", ["pooled"], ["flat"]),
             helper.make_node("MatMul", ["flat", "W"], ["logits"])]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 224, 224])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 5])],
                              initializer=[numpy_helper.from_array(w, "W")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    return path


def test_onnx_classifier_returns_probabilities(tiny_model):
    clf = OnnxClassifier(tiny_model)
    faces = [np.full((224, 224, 3), v, np.uint8) for v in (0, 128, 255)] * 12  # 36 images: more than one batch
    p = clf.predict_proba(faces, batch_size=16)
    assert p.shape == (36, 5) and p.dtype == np.float32 and np.allclose(p.sum(1), 1, atol=1e-6)
    assert clf.predict_proba([]).shape == (0, 5)


def test_brighter_red_channel_raises_the_first_class(tiny_model):
    clf = OnnxClassifier(tiny_model)
    dark, red = np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8)
    red[..., 0] = 255
    p = clf.predict_proba([dark, red])
    assert p[1, 0] > p[0, 0]


def test_backend_is_chosen_by_file_extension(tiny_model):
    assert isinstance(load_classifier(tiny_model), OnnxClassifier)


def test_api_serves_with_onnx_backend(tiny_model, monkeypatch):
    import cv2
    from fastapi.testclient import TestClient

    from app.api import create_app
    from hairify.face import Box, crop_box, pick_face
    from hairify.pipeline import Hairify

    class OneBox:
        def crop(self, rgb):
            b = pick_face([Box(10, 10, 100, 100)])
            return crop_box(rgb, b), b, 1

    client = TestClient(create_app(Hairify(OnnxClassifier(tiny_model), OneBox())))
    img = cv2.imencode(".jpg", np.full((200, 200, 3), 90, np.uint8))[1].tobytes()
    r = client.post("/analyse", content=img, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200 and abs(sum(r.json()["probabilities"].values()) - 1) < 1e-3
    assert client.get("/health").json()["backend"] == "OnnxClassifier"
