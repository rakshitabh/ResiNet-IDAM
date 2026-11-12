# Multi-stage production Dockerfile for ResiNet-IDAM v2.0
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight runtime
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r resinet && useradd -r -g resinet resinet

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/resinet/.local

# Copy application source, models, and configurations
COPY --chown=resinet:resinet src/ /app/src/
COPY --chown=resinet:resinet models/ /app/models/
COPY --chown=resinet:resinet data/processed/ /app/data/processed/
COPY --chown=resinet:resinet app.py /app/app.py

ENV PATH=/home/resinet/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

USER resinet

EXPOSE 8501 8000

# Default entrypoint starts Streamlit SecOps operations console
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

