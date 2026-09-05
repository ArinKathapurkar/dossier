FROM python:3.12-slim

# Build tools are needed for a couple of wheels; removed in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY dossier ./dossier
COPY prompts ./prompts
COPY tests ./tests

RUN pip install --no-cache-dir -e .

# The embedding and reranker models are NOT baked into the image: they download from the
# HuggingFace hub on the first search (~150 MB combined) into this cache directory. Mount
# a volume at /root/.cache/huggingface to make that a one-time cost across restarts.
ENV HF_HOME=/root/.cache/huggingface \
    DOSSIER_GRAPH_BACKEND=networkx \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "dossier.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
