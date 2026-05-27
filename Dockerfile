# ── Stage 1: Install Python deps ─────────────────────────────────────────────
FROM python:3.11-slim AS python-deps

WORKDIR /app
COPY ml-service/requirements.txt ./ml-service/requirements.txt
RUN pip install --no-cache-dir -r ml-service/requirements.txt

# ── Stage 2: Build Next.js app ────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 3: Production image ─────────────────────────────────────────────────
FROM node:20-slim AS runner

# Install Python 3 in the final image
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8080

# Copy Python site-packages from python-deps stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin/python3.11 /usr/local/bin/python3.11
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python3

# Copy Next.js standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Copy ML service (scripts + trained models)
COPY --from=builder /app/ml-service ./ml-service

EXPOSE 8080

CMD ["node", "server.js"]
