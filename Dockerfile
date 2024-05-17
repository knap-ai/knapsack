# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
# Python won’t try to write .pyc files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE 1
# Python outputs everything to terminal immediately and buffers it first
ENV PYTHONUNBUFFERED 1
ENV PYTHONHASHSEED 1024

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
# Ensure that Poetry is installed and paths are setup
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

# Copy the Poetry configuration files before the rest of the app to avoid unnecessary re-installs at each build
COPY pyproject.toml poetry.lock* /app/

# Install dependencies using Poetry
# Set virtualenvs.create false to install dependencies system-wide (important for containers)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY . /app

WORKDIR /app

CMD ["python", "-m", "knapsack.cli", "deploy", "--port",  "8888"]
