# ── Stage 1: Build Next.js frontend ──────────────────────────────────────────
FROM node:20-slim AS next-builder
WORKDIR /app
# Copy ONLY package.json — NOT package-lock.json.
# The lockfile was generated on Windows and contains Windows-native binary paths
# (lightningcss-win32, @next/swc-win32, etc.) that will crash on Linux.
# Without the lockfile, npm resolves fresh and downloads correct Linux binaries.
COPY package.json ./
# Install all deps including devDependencies needed for the build
RUN npm install --include=dev --legacy-peer-deps
COPY . .
# Increase Node heap size for large Next.js builds; build the app
RUN NODE_OPTIONS="--max-old-space-size=4096" npm run build

# ── Stage 2: Train ML Models ──────────────────────────────────────────────────
FROM python:3.11-slim AS ml-builder
WORKDIR /app
# Set Agg backend env var so matplotlib never tries to open a display
ENV MPLBACKEND=Agg
COPY ml-service/requirements.txt ./ml-service/requirements.txt
RUN pip install --no-cache-dir -r ml-service/requirements.txt
COPY ml-service ./ml-service
COPY upload ./upload
# Create outputs directory (needed by train_models.py for plot saving)
RUN mkdir -p /app/outputs
# Run the full ML training pipeline — generates all .pkl & .json model artifacts
RUN python ml-service/train_models.py

# ── Stage 3: Assemble Production Runner ───────────────────────────────────────
FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1
# Force headless matplotlib in production API calls too
ENV MPLBACKEND=Agg
# Ensure Python can find the site-packages copied from the ml-builder stage
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

# Install Python 3.11 runtime from deadsnakes or standard slim — use the same version as ml-builder
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from ml-builder into the runner's site-packages path
COPY --from=ml-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Create a 'python' alias for 'python3' so any legacy calls work
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy Next.js standalone server and static assets from next-builder stage
COPY --from=next-builder /app/.next/standalone ./
COPY --from=next-builder /app/.next/static ./.next/static
COPY --from=next-builder /app/public ./public

# Copy pre-trained ML models and all service scripts from ml-builder stage
COPY --from=ml-builder /app/ml-service ./ml-service

EXPOSE 8080
CMD ["node", "server.js"]
