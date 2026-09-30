# DriftForge AI: single container running the Express/React app and the Python FastAPI core.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Cloud Run, or any Docker host.
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000, which is the "node" user in this image
USER node
WORKDIR /home/node/app

# Python dependencies first so this layer is cached across code changes.
# server.ts starts the backend with backend/.venv automatically.
COPY --chown=node:node backend/requirements.txt backend/requirements.txt
RUN python3 -m venv backend/.venv \
    && backend/.venv/bin/pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=node:node package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY --chown=node:node . .
RUN npx vite build

ENV NODE_ENV=production \
    PORT=7860
EXPOSE 7860

CMD ["npm", "start"]
