# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy


# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install necessary libraries for dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*



ADD . /app

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --compile-bytecode

# Switch back to the main working directory

# Copy the source code into the container

# Expose the port that the application listens on
EXPOSE 8000

# Run the application
CMD uv run uvicorn app:app --host 0.0.0.0 --port 8000
