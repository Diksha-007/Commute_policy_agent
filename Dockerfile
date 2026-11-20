FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model (BAAI/bge-base-en-v1.5, about 210 MB) at build time, so the first request is not
# slow and the container works offline.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='BAAI/bge-base-en-v1.5', cache_dir='/opt/fastembed').embed(['warmup']))"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
