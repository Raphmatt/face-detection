# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run server
uv run uvicorn src.app:app --reload

# Run all tests
uv run pytest -q

# Run single test
uv run pytest src/tests/test_service.py::test_function_name -v

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Docker
docker compose up --build
```

## Architecture

FastAPI microservice that validates and optimizes profile photos. Validates face presence/pose, removes background, aligns to uniform template, returns transparent PNG.

**Processing Pipeline:**

```
Upload → Validation (face count, pose, shoulders) → Alignment (rotate/scale) → Segmentation (background removal) → PNG
```

**Key Modules:**

- `src/service.py` - Orchestration layer, calls validation then alignment
- `src/face_utilities.py` - Core CV logic: face detection, pose validation, segmentation (MediaPipe Tasks API + dlib fallback)
- `src/face_aligner.py` - `FaceAligner` class for rotation/scaling/cropping to template
- `src/router.py` - FastAPI routes (`/heartbeat`, `/image/process`)

**ML Models** (in `src/models/`):

- MediaPipe: FaceLandmarker, PoseLandmarker, ImageSegmenter (selfie/multiclass)
- dlib: shape_predictor_68_face_landmarks (fallback)

**Validation Rules:**

1. Exactly one face
2. Face angle < 10°
3. Face looking straight (MediaPipe pose estimation)
4. Shoulders level

## Standalone Scripts

`scripts/` contains PEP 723 standalone scripts runnable via `uv run`:

- `demo_webcam.py` - Real-time webcam face alignment
- `prepare_id_photo.py` - Swiss ID photo format preparation (Art. 12 VAwG)
