# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install necessary libraries for dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    bzip2 \
    g++ \
    graphviz \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Clone and build Dlib
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /usr/src/app/dlib
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .

WORKDIR /usr/src/app

# Install dependencies using uv and pyproject.toml
# Leverage a cache mount to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable

# Switch back to the main working directory

# Copy the source code into the container
COPY . .

# Expose the port that the application listens on
EXPOSE 8000

# Run the application
CMD cd src && uvicorn app:app --host 0.0.0.0 --port 8000
