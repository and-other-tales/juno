FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src ./src

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to not use virtualenvs in container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "othertales_juno"]