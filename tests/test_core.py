"""Face handling, recommendations, pipeline and API. Needs neither PyTorch nor ONNX Runtime."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hairify import CLASSES
from hairify.face import Box, InvalidImageError, NoFaceError, crop_box, decode_image, pick_face
from hairify.pipeline import Hairify
from hairify.preprocess import softmax, to_array
from hairify.recommend import knowledge_base, recommend, recommend_from_probabilities


def jpeg(h=300, w=300, color=(120, 140, 160)) -> bytes:
    return cv2.imencode(".jpg", np.full((h, w, 3), color, np.uint8))[1].tobytes()


# ---- face handling ---------------------------------------------------------------------------------------

def test_largest_face_is_chosen_not_the_first():
    # The original notebook used boxes[0]; Haar returns boxes in arbitrary order.
    neck, face = Box(100, 300, 70, 70), Box(80, 60, 200, 220)
    assert pick_face([neck, face]) == face
    with pytest.raises(NoFaceError):
        pick_face([])


def test_crop_keeps_margin_and_size():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[100:300, 100:300] = 255
    out = crop_box(rgb, Box(100, 100, 200, 200))
    assert out.shape == (224, 224, 3)
    assert out[0, 0].sum() == 0 and out[112, 112].sum() == 765  # 10% dark margin around the white face box


def test_crop_is_clipped_at_image_edges():
    assert crop_box(np.ones((100, 100, 3), np.uint8), Box(0, 0, 100, 100)).shape == (224, 224, 3)


@pytest.mark.parametrize("data, message", [(b"", "empty"), (b"not an image", "readable"), (jpeg(20, 20), "between")])
def test_invalid_uploads_are_rejected(data, message):
    with pytest.raises(InvalidImageError, match=message):
        decode_image(data)


def test_valid_upload_decodes_to_rgb():
    rgb = decode_image(jpeg(color=(0, 0, 255)))  # BGR red in OpenCV order
    assert rgb.shape == (300, 300, 3) and rgb[0, 0, 0] > 200


def test_preprocessing_matches_imagenet_normalisation():
    x = to_array([np.zeros((224, 224, 3), np.uint8)] * 2)
    assert x.shape == (2, 3, 224, 224) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.isclose(x[0, 0, 0, 0], -0.485 / 0.229)
    p = softmax(np.array([[1.0, 2.0, 3.0, 0.0, -1.0]]))
    assert np.isclose(p.sum(), 1) and p.argmax() == 2


# ---- recommendations -------------------------------------------------------------------------------------

def test_knowledge_base_covers_every_class():
    db = knowledge_base()
    assert set(db) == set(CLASSES)
    assert all(len(v["suited_styles"]) == 5 and v["avoid"] for v in db.values())


def test_recommendations_are_sorted_and_unknown_shape_is_an_error():
    scores = [s["score"] for s in recommend("Round", 3)["recommended_styles"]]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)
    with pytest.raises(ValueError):
        recommend("Triangle")


def test_close_call_adds_second_shape():
    confident = recommend_from_probabilities([0.05, 0.05, 0.8, 0.05, 0.05])
    assert confident["face_shape"] == "Oval" and not confident["close_call"] and "also_consider" not in confident
    close = recommend_from_probabilities([0.05, 0.05, 0.42, 0.40, 0.08])
    assert close["close_call"] and close["also_consider"]["face_shape"] == "Round"


# ---- pipeline and API with a fake detector and classifier --------------------------------------------------

class FixedCropper:
    def __init__(self, boxes):
        self.boxes = boxes

    def crop(self, rgb):
        box = pick_face(self.boxes)
        return crop_box(rgb, box), box, len(self.boxes)


class AlwaysSquare:
    def predict_proba(self, faces):
        p = np.full((len(faces), 5), 0.02, np.float32)
        p[:, CLASSES.index("Square")] = 0.92
        return p


def engine(boxes=(Box(50, 50, 150, 150),)):
    return Hairify(AlwaysSquare(), FixedCropper(list(boxes)))


def test_pipeline_end_to_end():
    res = engine([Box(10, 10, 40, 40), Box(50, 50, 150, 150)]).analyse(jpeg())
    assert res["face_shape"] == "Square" and res["faces_detected"] == 2
    assert res["face_box"] == {"x": 50, "y": 50, "width": 150, "height": 150}
    assert set(res["timing_ms"]) == {"decode", "face_detection", "classification", "total"}


@pytest.fixture
def client():
    from app.api import create_app

    return TestClient(create_app(engine()))


def test_api_analyse_and_health(client):
    r = client.post("/analyse", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200 and r.json()["face_shape"] == "Square"
    health = client.get("/health").json()
    assert health["model_loaded"] is True and health["backend"] == "AlwaysSquare"
    assert client.get("/").status_code == 200


def test_api_errors(client):
    assert client.post("/analyse", content=b"x", headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post("/analyse", content=b"garbage", headers={"Content-Type": "image/png"}).status_code == 400
    assert client.post("/analyse?top_n=9", content=jpeg(), headers={"Content-Type": "image/jpeg"}).status_code == 422
    from app.api import create_app

    r = TestClient(create_app(engine(boxes=()))).post("/analyse", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 422 and "no face" in r.json()["detail"]


def test_missing_weights_returns_503(monkeypatch, tmp_path):
    from app.api import create_app

    monkeypatch.setenv("HAIRIFY_WEIGHTS", str(tmp_path / "missing.onnx"))
    r = TestClient(create_app()).post("/analyse", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 503 and "not found" in r.json()["detail"]
