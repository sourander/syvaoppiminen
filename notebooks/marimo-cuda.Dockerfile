# Base image with uv + Python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system environment
RUN uv sync --frozen --no-dev

# Expose Marimo default port
EXPOSE 2718

# Run Marimo
CMD ["marimo", "run", "--host", "0.0.0.0", "--port", "2718", "--no-token"]
