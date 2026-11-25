"""HTTP service.

    HAIRIFY_WEIGHTS=models/face_shape_classifier.onnx uvicorn app.api:app --port 8000   # or a .pt checkpoint

POST /analyse with the photo as the raw request body (Content-Type image/jpeg, image/png, ...).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from hairify import __version__
from hairify.face import MAX_BYTES, InvalidImageError, NoFaceError
from hairify.onnx_backend import load_classifier
from hairify.pipeline import Hairify

log = logging.getLogger("hairify")
STATIC = Path(__file__).with_name("static")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def create_app(engine: Hairify | None = None) -> FastAPI:
    app = FastAPI(title="Hairify", version=__version__,
                  description="Face-shape classification and hairstyle recommendations")
    state = {"engine": engine, "started": time.time()}

    def get_engine() -> Hairify:
        if state["engine"] is None:
            weights = os.environ.get("HAIRIFY_WEIGHTS", "models/face_shape_classifier.onnx")
            if not Path(weights).is_file():
                raise HTTPException(503, f"model weights not found at {weights}; set HAIRIFY_WEIGHTS")
            state["engine"] = Hairify(load_classifier(weights))
        return state["engine"]

    @app.get("/health")
    def health():
        engine = state["engine"]
        backend = type(engine.classifier).__name__ if engine is not None else None
        return {"status": "ok", "version": __version__, "model_loaded": engine is not None, "backend": backend,
                "uptime_seconds": round(time.time() - state["started"], 1)}

    @app.post("/analyse")
    async def analyse(request: Request, top_n: int = 5):
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_TYPES:
            raise HTTPException(415, f"send the photo as the request body with Content-Type one of {sorted(ALLOWED_TYPES)}")
        if not 1 <= top_n <= 5:
            raise HTTPException(422, "top_n must be between 1 and 5")
        length = request.headers.get("content-length")
        if length and int(length) > MAX_BYTES:
            raise HTTPException(413, "image too large (10 MB maximum)")
        body = await request.body()
        try:
            result = get_engine().analyse(body, top_n)
        except InvalidImageError as e:
            raise HTTPException(400, str(e)) from e
        except NoFaceError as e:
            raise HTTPException(422, str(e)) from e
        log.info("analysed image: shape=%s confidence=%.2f total_ms=%.0f", result["face_shape"],
                 result["confidence"], result["timing_ms"]["total"])
        return result

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
