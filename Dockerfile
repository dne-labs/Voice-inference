# syntax=docker/dockerfile:1

# VoiceForge runs as three processes. This file defines one build target per
# process so docker compose can build the whole stack from a single Dockerfile.

########## Inference worker ##########
FROM python:3.12-slim AS worker

# Override these two arguments together to build a CUDA worker, for example:
#   --build-arg TORCH_VARIANT=cu126
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.8.0
ARG TORCH_VARIANT=cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG UV_VERSION=0.13.9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/voiceforge/.cache/huggingface \
    INFERENCE_HOST=0.0.0.0 \
    INFERENCE_PORT=8000 \
    PATH="/app/services/inference/.venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl espeak-ng \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "uv==${UV_VERSION}" \
 && useradd --create-home --uid 10001 voiceforge \
 && mkdir -p /app/services/inference /home/voiceforge/.cache/huggingface \
 && chown -R voiceforge:voiceforge /app /home/voiceforge

WORKDIR /app/services/inference
COPY --chown=voiceforge:voiceforge services/inference/pyproject.toml services/inference/uv.lock ./
USER voiceforge

# Match the runtime pinned by the README's uv commands. The project itself has
# no dependencies, so the model runtime is installed into the same virtualenv.
RUN uv sync --locked --no-dev --no-install-project --python 3.12
RUN uv pip install --python .venv/bin/python --index "${TORCH_INDEX}" \
      "torch==${TORCH_VERSION}+${TORCH_VARIANT}" "kokoro==0.9.4"

COPY --chown=voiceforge:voiceforge services/inference/src ./src
RUN uv pip install --python .venv/bin/python --no-deps .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["voiceforge-worker", "--device", "cpu"]

########## API gateway ##########
FROM node:22-slim AS gateway

ENV NODE_ENV=production
WORKDIR /app/apps/api

COPY apps/api/package.json apps/api/package-lock.json ./
RUN npm ci --omit=dev

COPY apps/api/server.js ./
# The benchmark routes read the recorded reports relative to the server.
COPY experiments /app/experiments

ENV HOST=0.0.0.0 \
    PORT=3001
USER node
EXPOSE 3001
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:3001/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]

########## Next.js web app ##########
FROM node:22-slim AS web

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app
RUN chown node:node /app

COPY --chown=node:node apps/web/package.json apps/web/package-lock.json ./
USER node
RUN npm ci

COPY --chown=node:node apps/web ./
# Next.js bakes rewrite destinations into the build, so the gateway address
# must be known here. It is the address used by the Next.js server, not the
# browser.
ARG VOICEFORGE_API_URL=http://127.0.0.1:3001
ENV VOICEFORGE_API_URL=${VOICEFORGE_API_URL}
RUN npm run build

ENV NODE_ENV=production
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["./node_modules/.bin/next", "start", "--hostname", "0.0.0.0", "--port", "3000"]
