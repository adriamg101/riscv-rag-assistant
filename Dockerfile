FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app/ app/
COPY frontend/ frontend/
COPY data/corpus/ data/corpus/

RUN useradd --create-home appuser \
    && mkdir -p chroma_db .cache/huggingface \
    && chown -R appuser:appuser /app/chroma_db /app/.cache
USER appuser

EXPOSE 8000 7860

# Ingestion is idempotent, so later starts recompute nothing.
CMD ["sh", "-c", "python -m app.ingest && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
