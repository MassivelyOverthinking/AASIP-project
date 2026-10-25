
# Use a lightweight Python base image
FROM python:3.12-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure Python and uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

# Install dependencies first to improve build caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Copy application source, templates and data
COPY src/ ./src/

# Flask listens on port 8001
EXPOSE 8001

# Run the application as a Python module
CMD ["python", "-m", "titanic_project.main"]


