# RoleRAG app image (FastAPI + Angular /app SPA + legacy /play UI + CLI).
#
# The model server is deliberately NOT containerized: on Apple Silicon, Docker has
# no GPU passthrough, so a local LLM in a container runs CPU-only and is unusably
# slow for the recommended 26B model. Run llama-server on the host (port 8080); this
# container reaches it via host.docker.internal. See README "Run in Docker".

# --- Frontend build stage: produce the Angular SPA bundle served at /app ---
FROM node:20-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npx ng build --base-href=/app/

# --- App image ---
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

# SPA build output — app.main mounts this at /app (path is <repo>/frontend/dist/frontend/browser).
COPY --from=frontend /build/frontend/dist/frontend/browser ./frontend/dist/frontend/browser

# Container-internal defaults; override the URLs/mode in docker-compose.yml.
ENV DATABASE_PATH=data/rolerag.db
EXPOSE 8000

# Bind 0.0.0.0 so the port is reachable from the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
