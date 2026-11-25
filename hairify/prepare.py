"""Re-crop the FaceShape dataset with the fixed (largest-face) cropper, keeping a given train/val/test split.

    python -m hairify.prepare --raw "FaceShape Dataset/training_set" --split-from old_split --out data/clean \
        --exclude exclude.txt

`--split-from` points to a folder laid out as <split>/<Class>/<name>.jpg; each raw image goes to the split its
file stem was in, so results stay comparable with earlier experiments. Images listed in `--exclude`
(one "<Class>/<file>" per line) are left out, e.g. training images that duplicate official test images.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from . import CLASSES
from .face import FaceCropper, NoFaceError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_membership(split_root: Path) -> dict:
    member = {}
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            for f in (split_root / split / cls).glob("*.jpg"):
                member[(cls, f.stem)] = split
    return member


def prepare(raw: Path, split_root: Path, out: Path, exclude: set[str]) -> dict:
    member = split_membership(split_root)
    cropper = FaceCropper()
    stats = Counter()
    for cls in CLASSES:
        for f in sorted((raw / cls).iterdir()):
            if f.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if f"{cls}/{f.name}" in exclude:
                stats["excluded"] += 1; continue
            split = member.get((cls, f.stem))
            if split is None:
                stats["not_in_original_split"] += 1; continue
            img = cv2.imread(str(f))
            if img is None:
                stats["unreadable"] += 1; continue
            try:
                face, _, n = cropper.crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except NoFaceError:
                stats["no_face"] += 1; continue
            stats["several_detections"] += n > 1
            dst = out / split / cls / (f.stem + ".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats[f"saved_{split}"] += 1
    return dict(stats)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True)
    p.add_argument("--split-from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", default=None)
    args = p.parse_args(argv)
    exclude = set()
    if args.exclude:
        exclude = {line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()}
    stats = prepare(Path(args.raw), Path(args.split_from), Path(args.out), exclude)
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    main()
