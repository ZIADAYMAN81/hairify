"""Hairstyle recommendations from the curated knowledge base, aware of how confident the classifier is."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import CLASSES

DB_PATH = Path(__file__).with_name("data") / "hairstyles.json"
CLOSE_CALL = 0.15  # when the top two shapes are within 15 points, show styles that suit both


@lru_cache(maxsize=1)
def knowledge_base() -> dict:
    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    missing = set(CLASSES) - set(db)
    if missing:
        raise RuntimeError(f"knowledge base is missing face shapes: {sorted(missing)}")
    return db


def recommend(shape: str, top_n: int = 5) -> dict:
    db = knowledge_base()
    if shape not in db:  # the notebook silently fell back to "Oval" here
        raise ValueError(f"unknown face shape {shape!r}; expected one of {list(db)}")
    entry = db[shape]
    styles = sorted(entry["suited_styles"], key=lambda s: -s["score"])[:top_n]
    return {"face_shape": shape, "description": entry["description"], "recommended_styles": styles,
            "styles_to_avoid": entry["avoid"]}


def recommend_from_probabilities(probs, top_n: int = 5) -> dict:
    """Use the predicted shape, but when the second shape is almost as likely, add its best styles instead of
    presenting an uncertain prediction as certain."""
    ranked = sorted(zip(CLASSES, map(float, probs)), key=lambda t: -t[1])
    (first, p1), (second, p2) = ranked[0], ranked[1]
    rec = recommend(first, top_n)
    rec["confidence"] = round(p1, 4)
    rec["probabilities"] = {c: round(p, 4) for c, p in ranked}
    rec["close_call"] = (p1 - p2) < CLOSE_CALL
    if rec["close_call"]:
        rec["also_consider"] = {"face_shape": second, "styles": recommend(second, 2)["recommended_styles"]}
    return rec
