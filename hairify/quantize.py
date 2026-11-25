"""Static int8 quantisation of the ONNX model with ONNX Runtime, calibrated on validation crops.

    python -m hairify.quantize --model models/face_shape_classifier.onnx --calibration data/clean/val \
        --out models/face_shape_classifier.int8.onnx

Calibration uses the validation split only (never the test set). Weights are quantised per channel; activations
use min-max ranges observed on the calibration images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import CLASSES
from .preprocess import to_array


class FolderReader:
    """Feeds pre-cropped calibration images to the quantiser in batches."""

    def __init__(self, root: Path, input_name: str, per_class: int = 40, batch: int = 8):
        files = [f for c in CLASSES for f in sorted((root / c).glob("*.jpg"))[:per_class]]
        faces = [cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), (224, 224)) for f in files]
        self.batches = iter([{input_name: to_array(faces[i:i + batch])} for i in range(0, len(faces), batch)])
        self.count = len(faces)

    def get_next(self):
        return next(self.batches, None)


def quantize(model: Path, calibration: Path, out: Path, per_class: int = 40, method_name: str = "minmax") -> dict:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[method_name]
    prepared = out.with_suffix(".prep.onnx")
    quant_pre_process(str(model), str(prepared), skip_symbolic_shape=True)  # input shape is fixed except batch
    input_name = ort.InferenceSession(str(prepared), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    reader = FolderReader(calibration, input_name, per_class)
    # Quantise only the convolutional backbone. Quantising the small classifier head (Gemm + BatchNorm1d)
    # destroyed accuracy (18.6%, chance level) in the first attempt, and the head costs almost no time anyway.
    quantize_static(str(prepared), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=method, op_types_to_quantize=["Conv"])
    prepared.unlink(missing_ok=True)
    info = {"calibration_images": reader.count, "fp32_mb": round(model.stat().st_size / 1e6, 1),
            "int8_mb": round(out.stat().st_size / 1e6, 1)}
    print(json.dumps(info))
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", required=True, help="validation crops, one folder per class")
    p.add_argument("--out", required=True)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax")
    args = p.parse_args(argv)
    quantize(Path(args.model), Path(args.calibration), Path(args.out), args.per_class, args.method)


if __name__ == "__main__":
    main()
