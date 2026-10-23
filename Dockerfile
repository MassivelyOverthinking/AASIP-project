FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory inside the container
WORKDIR /app

# Configure UV
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy dependency definitions first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the project itself
RUN uv sync --frozen 

# Copy the application source code
COPY . .

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Document the port used by Flask
EXPOSE 5000

# Run the Flask application
CMD ["python", "-m", "flask", "--app", "titanic_project.main", "run", "--host=0.0.0.0", "--port=5000"]
