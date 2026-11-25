# CPU image for the Hairify API. The model weights are not baked in: mount them at /models.
#   docker build -t hairify .
#   docker run -p 8000:8000 -v /path/to/models:/models hairify
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HAIRIFY_WEIGHTS=/models/face_shape_classifier.pt

# OpenCV needs these shared libraries even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only PyTorch keeps the image far smaller than the CUDA build
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY hairify/ hairify/
COPY app/ app/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
