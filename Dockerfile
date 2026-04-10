FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Default config (overridden by volume mount)
COPY config/ config/
COPY scripts/ scripts/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

VOLUME ["/app/data", "/app/config"]

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD pgrep -f "mindsecretary.app" > /dev/null || exit 1

CMD ["python", "-m", "mindsecretary.app"]
