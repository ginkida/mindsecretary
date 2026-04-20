FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MINDSECRETARY_ROOT=/app

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Default config (overridden by volume mount). migrations/ is source-of-truth,
# not user-editable — baked into the image, not bind-mounted.
COPY config/ config/
COPY scripts/ scripts/
COPY migrations/ migrations/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

VOLUME ["/app/data", "/app/config"]

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python scripts/healthcheck.py

CMD ["python", "-m", "mindsecretary.app"]
