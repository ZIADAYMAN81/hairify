# Hairify: Face-Shape Classification and Hairstyle Recommendation

> Graduation project, BSc Computer Science (Artificial Intelligence), The British University in Egypt, 2026. Author: Ziad Ayman.

Hairify classifies a person's **face shape** from a photo (Heart, Oblong, Oval, Round or Square) with a fine-tuned EfficientNet-B3 and recommends hairstyles from a curated styling knowledge base. It is served as a **FastAPI service with a web page**, runs on **ONNX Runtime in a 941 MB CPU Docker image**, and every number below is reproducible with the commands in this README.

## Results

Measured on the dataset's **official test set**: 996 face images never used for training, model selection or tuning. Intervals are 95% bootstrap intervals.

| Model | Accuracy | Weighted F1 | Macro ROC-AUC |
|---|---|---|---|
| Graduation-project model, original face cropper | 69.8% | 0.696 | 0.914 |
| Graduation-project model, corrected face cropper | 73.0% (70.3–75.5) | 0.729 | 0.931 (0.921–0.941) |
| **Released model** (served on CPU with ONNX Runtime) | **79.2%** (76.8–81.6) | **0.792** | **0.947** (0.937–0.957) |

The released model scores 79.5% when evaluated on a GPU. Four test images are near-ties (the top two shapes are within 0.2 percentage points), and GPU and CPU arithmetic break those ties differently. The ONNX model reproduces PyTorch on the CPU exactly: same predictions on all 996 images, logits within 6e-6.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Heart | 0.82 | 0.80 | 0.81 |
| Oblong | 0.89 | 0.84 | 0.87 |
| Oval | 0.75 | 0.63 | 0.68 |
| Round | 0.69 | 0.85 | 0.77 |
| Square | 0.83 | 0.84 | 0.83 |

Oval is the hardest class: it sits between the others by definition and is most often confused with Round.

**End-to-end check:** all 1,000 official test photos were sent to the running Docker container over HTTP. It returned the same 79.2% accuracy, answered "no face found" (HTTP 422) for the 4 images without a detectable face, flagged 98 predictions as close calls, and took a median of 65 ms per request.

### What changed, and what made the difference

Each step below was trained once and compared on the same 996 images with a paired McNemar test (all evaluated on the GPU):

| Step | Accuracy | Change | Images fixed / broken | p-value |
|---|---|---|---|---|
| Graduation-project model | 73.0% | | | |
| Retrained on corrected face crops, same recipe | 74.4% | +1.4 points | 67 / 53 | 0.24 (not significant) |
| Learning-rate schedule in phase 2: plateau instead of cosine | 78.6% | **+4.2 points** | 74 / 32 | **0.00006** |
| Stronger augmentation, weight decay 0.05, early stopping | 79.5% | +0.9 points | 75 / 66 | 0.50 (not significant) |
| **All changes together** | | **+6.5 points** | 122 / 57 | **0.000001** |

The learning-rate schedule made the largest difference: cosine annealing lowers the rate on a fixed 60-epoch timetable, while the plateau schedule halves it only when validation accuracy stops improving. The corrected face cropper matters most at prediction time (69.8% to 73.0% for the same model). Caveat: each setting was trained with a single random seed; the McNemar test covers test-set sampling, not run-to-run training variance.

## What the audit found

1. **A face-cropping bug.** OpenCV's Haar detector returns boxes in no particular order and the original code used the first one. When it found several (12.5% of training images, 13% of test images) it could crop a neck or background patch; one such crop sat in the test set labelled "Heart". The cropper now uses the largest detection (`hairify/face.py`).
2. **The official test set had never been used.** Earlier results came from a split of the training folder; the official test set gives the first fully independent estimate.
3. **The reported accuracy was not reproducible.** The dissertation reported 72.8% on the internal 600-image split; re-running the saved model gives 72.3%. The released model scores 81.0% on that split.
4. **Data checks.** A perceptual-hash scan of all 5,000 images found 8 test images duplicated in the training folder and one image filed under two classes. These 10 were excluded from retraining.
5. **RetinaFace never ran.** It needs TensorFlow, which was not installed, so every image used the Haar fallback. Training and serving now deliberately use the same Haar detector, and OpenCV is pinned to the training version (4.13.0.92), so production crops match the crops the model learned from.

## How it works

```
photo -> validation -> Haar face detection (largest face, 10% margin) -> EfficientNet-B3 (ONNX) -> 5 probabilities
                                                                                       |
                     recommendations (25 styles, "avoid" list; blends the top two shapes on close calls)
```

- **Model:** ImageNet-pretrained EfficientNet-B3 (`timm`) with a head `1536 -> 512 -> BN -> ReLU -> Dropout(0.4) -> 256 -> ReLU -> Dropout(0.2) -> 5`, trained in two phases: first the head on the frozen backbone, then the top five blocks fine-tuned.
- **Serving:** the PyTorch checkpoint is exported to ONNX (opset 17) and served with ONNX Runtime; the serving code and image contain no PyTorch.
- **Confidence-aware recommendations:** when the top two shapes are within 15 points, the response marks a close call and includes styles for both.

Diagrams, error handling and design decisions are in [docs/architecture.md](docs/architecture.md).

## Performance

Laptop with an AMD Ryzen CPU and an RTX 3060 Laptop GPU (6 GB). Times include converting images to model input:

| Backend | Model, one image | Full request (decode, face detection, model) | Batch of 32 |
|---|---|---|---|
| **ONNX Runtime, CPU** (served) | **12 ms** | **19 ms** | 38 images/s |
| PyTorch, CPU | 34 ms | 40 ms | 34 images/s |
| PyTorch, GPU | 21 ms | 26 ms | 364 images/s |

**Deployment decision.** Hairify handles one photo per request. For that, ONNX Runtime on a CPU is faster than PyTorch on the GPU, the image is 941 MB instead of 10.7 GB with CUDA PyTorch, and it runs on any machine or free hosting tier. A GPU image would only pay off for bulk batch processing.

**Int8 quantisation was tried and rejected.** Static post-training quantisation with ONNX Runtime (four configurations: per-channel and per-tensor weights, convolution-only, two activation types; calibrated on validation images) reduced accuracy to chance (about 20%). EfficientNet's depthwise convolutions, squeeze-and-excitation blocks and SiLU activations are known to be difficult to quantise after training; quantisation-aware training would be the next step. `hairify/quantize.py` is kept so the result can be reproduced.

## Run it

Serve (ONNX Runtime, no PyTorch needed):
```bash
pip install -r requirements-serve.txt
# put the exported model at models/face_shape_classifier.onnx
uvicorn app.api:app --port 8000          # web page at http://localhost:8000
curl -X POST "http://localhost:8000/analyse" -H "Content-Type: image/jpeg" --data-binary @photo.jpg
```

Docker:
```bash
docker build -t hairify .
docker run -p 8000:8000 -v "$PWD/models:/models:ro" hairify
```

Train, export and evaluate (PyTorch):
```bash
pip install -r requirements.txt
python -m hairify.prepare  --raw "FaceShape Dataset/training_set" --split-from <split> --out data/clean
python -m hairify.train    --data data/clean --out models/run --recipe regularised
python -m hairify.export   --weights models/run/face_shape_classifier.pt --out models/face_shape_classifier.onnx
python -m hairify.evaluate --weights models/face_shape_classifier.onnx --data "FaceShape Dataset/testing_set"
python scripts/benchmark.py --weights models/face_shape_classifier.onnx --image photo.jpg
python -m pytest -q        # core tests always run; PyTorch and ONNX tests run where those libraries are installed
```

## Limitations

- **Still overfits:** the released model reaches 99.7% training accuracy against 82.5% validation accuracy.
- **About one face in five is misclassified,** and face shape is partly subjective, so the labels themselves are noisy. Probabilities are returned with every prediction.
- **Dataset bias:** the Face Shape Dataset is mostly photos of female celebrities in studio lighting; accuracy on other groups and on phone photos has not been measured.
- **Haar detection is basic:** it misses some faces (4 of 1,000 test images) and assumes a roughly frontal face.
- **The weights are not in the repository** (46 MB). The dataset must be downloaded separately and its licence respected.
- **The notebook's virtual try-on is a colour overlay,** not generative hairstyle transfer; the service returns recommendations only.

## Original notebook

`notebooks/hairify.ipynb` is the graduation-project notebook: baselines (KNN, SVM, Random Forest, XGBoost and a stacking ensemble on identity embeddings, backbone and geometric features), Optuna search, Grad-CAM, t-SNE, and a celebrity-similarity experiment that failed (retrieval mAP 0.017, because identity embeddings do not encode face shape). Figures containing real faces are not published.

## Tech stack

Python · PyTorch · timm · ONNX · ONNX Runtime · OpenCV · FastAPI · scikit-learn · Docker · pytest
