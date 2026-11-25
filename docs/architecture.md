# Architecture

## Request path (serving)

```mermaid
flowchart LR
    U[Browser page<br/>app/static/index.html] -- "POST /analyse<br/>raw image bytes" --> A[FastAPI<br/>app/api.py]
    A --> V[decode_image<br/>type and size checks]
    V --> D[FaceCropper<br/>Haar detection,<br/>largest face, 10% margin]
    D --> M[Classifier<br/>EfficientNet-B3 + MLP head<br/>softmax over 5 shapes]
    M --> R[Recommender<br/>knowledge base of 25 styles,<br/>blends top two shapes on close calls]
    R -- "JSON: shape, probabilities,<br/>styles, face box, timings" --> U
    V -. "400 invalid image" .-> U
    D -. "422 no face" .-> U
```

| Stage | Module | Failure handling |
|---|---|---|
| Upload | `app/api.py` | Content type must be an image (415); body at most 10 MB (413); `top_n` 1–5 (422) |
| Decode | `hairify/face.py::decode_image` | Unreadable or tiny/huge images raise `InvalidImageError` → 400 |
| Face crop | `hairify/face.py::FaceCropper` | No face raises `NoFaceError` → 422; several faces → the largest one |
| Classify | `hairify/model.py::Classifier` | Weights loaded once at the first request; missing weights → 503 |
| Recommend | `hairify/recommend.py` | Unknown shape raises an error instead of silently defaulting |

## Training and evaluation path

```mermaid
flowchart LR
    RAW[FaceShape dataset<br/>training_set 4,000 images] --> P[hairify.prepare<br/>largest-face crops,<br/>original split kept,<br/>duplicates excluded]
    P --> T[hairify.train<br/>phase 1: head on frozen backbone<br/>phase 2: top 5 blocks unfrozen]
    T --> W[(face_shape_classifier.pt)]
    W --> E[hairify.evaluate<br/>accuracy, F1, AUC,<br/>bootstrap 95% intervals]
    OFF[Official testing_set<br/>1,001 images, never used for<br/>training or model selection] --> E
```

## Design decisions

- **The largest detection is the face.** OpenCV's Haar detector returns boxes in no order; the original notebook used the first one and sometimes cropped necks or background. 13% of test images and 12.5% of training images had several detections.
- **Haar instead of RetinaFace.** RetinaFace was the intended first choice, but it depends on TensorFlow, which was never installed, so every image used Haar. Serving uses the same detector as training so that the crops match.
- **The image travels as the raw request body**, not a multipart form, which avoids an extra dependency and makes the API easy to call with `curl --data-binary`.
- **Uncertainty is shown, not hidden.** When the top two shapes are within 15 points, the response sets `close_call` and adds styles for the second shape.
- **Weights are loaded lazily and are not in the image**, so the container stays small and the model can be swapped without rebuilding.
