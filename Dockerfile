# RoleRAG app image (FastAPI + /play UI + CLI).
#
# The model server is deliberately NOT containerized: on Apple Silicon, Docker has
# no GPU passthrough, so a local LLM in a container runs CPU-only and is unusably
# slow for the recommended 26B model. Run llama-server on the host (port 8080); this
# container reaches it via host.docker.internal. See README "Run in Docker".
FROM python:3.12-slim

# onnxruntime (pulled in by qdrant-client[fastembed]) needs libgomp at runtime;
# curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY data ./data
RUN pip install --no-cache-dir .

# Container-internal defaults; override the URLs/mode in docker-compose.yml.
ENV DATABASE_PATH=data/rolerag.db
EXPOSE 8000

# Bind 0.0.0.0 so the port is reachable from the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
