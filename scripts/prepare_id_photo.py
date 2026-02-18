#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mediapipe>=0.10.32",
#   "opencv-python-headless>=4.8.0",
#   "numpy<2",
#   "pillow>=12.1.1",
# ]
# ///

import argparse
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    print(f"Failed to import mediapipe: {e}", file=sys.stderr)
    sys.exit(1)


@dataclass
class StandardSpec:
    width: int  # Final image width in pixels
    height: int  # Final image height in pixels
    eye_y_from_bottom_pct: float  # Eyes' horizontal line as % of final image height from bottom (e.g., 55.0)
    pupil_dist_pct_of_width: float  # Pupil-to-pupil distance as % of final image width
    horiz_center_pct: float = 50.0  # Where to place eye midpoint horizontally in final image (0-100), default center

    def clamp(self) -> "StandardSpec":
        self.eye_y_from_bottom_pct = float(np.clip(self.eye_y_from_bottom_pct, 0, 100))
        self.pupil_dist_pct_of_width = float(
            np.clip(self.pupil_dist_pct_of_width, 1, 99)
        )
        self.horiz_center_pct = float(np.clip(self.horiz_center_pct, 0, 100))
        return self


PRESETS = {
    # Swiss ID photo format per Art. 12 VAwG (SR 143.11)
    # https://www.fedlex.admin.ch/eli/cc/2010/96/de#art_12
    # - 1980 × 1440 px (H × W)
    # - Pupil distance: 15–20% of image width
    # - Eyes: 50–60% of image height from bottom
    # - JPEG ~700 kB, high quality, baseline format
    "ch-id-2025": StandardSpec(
        width=1440,
        height=1980,
        eye_y_from_bottom_pct=55.0,
        pupil_dist_pct_of_width=18.0,
        horiz_center_pct=50.0,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face image cropper & aligner (MediaPipe-based). "
        "Detects a face, levels the eyes horizontally, centers, scales to a standard, and exports JPEG.",
        epilog=(
            "Examples:\n"
            "  # Simple: write next to input with a default name\n"
            "  ./script.py photo.jpg\n\n"
            "  # Explicit output path\n"
            "  ./script.py photo.jpg --output ./out/portrait_prepared.jpg\n\n"
            "  # Use built-in Swiss preset but with eyes at 58%% height-from-bottom and 20%% pupil distance\n"
            "  ./script.py photo.jpg --preset ch-id-2025 --eyes-y-bottom 58 --pupil-dist 20\n\n"
            "  # Define a custom standard (e.g., 1024x1280, eyes 60%% from bottom, pupil distance 19%%, eye center at 48%% width)\n"
            "  ./script.py photo.jpg --width 1024 --height 1280 --eyes-y-bottom 60 --pupil-dist 19 --h-center 48\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input image (JPEG/PNG/etc.). Required.",
    )
    parser.add_argument(
        "--output",
        help="Full output file path (overrides --out-dir/--out-name). Defaults to beside input.",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory for output (if --output not given). Default: input's directory.",
    )
    parser.add_argument(
        "--out-name",
        help="Output filename (if --output not given). Default: <stem>.prepared.jpg",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output without prompting.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="ch-id-2025",
        help="Standards preset to use. Default: ch-id-2025",
    )
    parser.add_argument(
        "--width", type=int, help="Final width in px (overrides preset)."
    )
    parser.add_argument(
        "--height", type=int, help="Final height in px (overrides preset)."
    )
    parser.add_argument(
        "--eyes-y-bottom",
        type=float,
        help="Eyes level as %% of image height measured from bottom (overrides preset). Example: 55",
    )
    parser.add_argument(
        "--pupil-dist",
        type=float,
        help="Pupil distance as %% of image width (overrides preset). Example: 18",
    )
    parser.add_argument(
        "--h-center",
        type=float,
        help="Horizontal placement of eye midpoint as %% of width (0=left, 50=center).",
    )
    parser.add_argument(
        "--target-size-kb",
        type=int,
        default=700,
        help="Target JPEG size in kilobytes (approx). Default: 700",
    )
    parser.add_argument(
        "--size-tolerance-kb",
        type=int,
        default=60,
        help="Allowed ±KB around target size. Default: 60",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=30.0,
        help="If absolute eye tilt exceeds this, rotation is still applied but a warning is printed. Default: 30",
    )
    return parser


def resolve_output_path(inp: Path, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name if args.out_name else f"{inp.stem}.prepared.jpg"
    return out_dir / out_name


def prompt_overwrite(path: Path) -> Optional[Path]:
    while True:
        choice = (
            input(f"Output '{path}' exists. [O]verwrite, [R]ename, [C]ancel? ")
            .strip()
            .lower()
        )
        if choice in ("o", "overwrite"):
            return path
        if choice in ("c", "cancel"):
            print("Cancelled.")
            return None
        if choice in ("r", "rename"):
            stem, suffix = path.stem, path.suffix or ".jpg"
            for i in range(1, 10000):
                candidate = path.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    print(f"Using '{candidate}'.")
                    return candidate
            print("Unable to pick a new name. Cancelled.")
            return None
        print("Please type O, R, or C.")


def load_image_bgr(path: Path) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"Failed to read image: {path}")
    return data


def save_jpeg_with_target_size(
    rgb_arr: np.ndarray, out_path: Path, target_kb: int, tol_kb: int
) -> None:
    img = Image.fromarray(rgb_arr, mode="RGB")
    low_q, high_q = 65, 95  # reasonable high-quality range
    best_q = high_q
    target_bytes = target_kb * 1024
    tol_bytes = tol_kb * 1024
    best_buf = None

    # Quick try at high quality
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=best_q,
        optimize=True,
        progressive=False,
        subsampling=0,
    )
    size = buf.tell()
    if abs(size - target_bytes) <= tol_bytes or size <= target_bytes + tol_bytes:
        best_buf = buf
    else:
        # Binary search to approach target size from above without going too low
        for _ in range(10):
            if size > target_bytes + tol_bytes:
                high_q = best_q - 1
            else:
                low_q = best_q + 1
            if low_q > high_q:
                break
            best_q = (low_q + high_q) // 2
            buf = io.BytesIO()
            img.save(
                buf,
                format="JPEG",
                quality=best_q,
                optimize=True,
                progressive=False,
                subsampling=0,
            )
            size = buf.tell()
            best_buf = buf

    if best_buf is None:
        best_buf = buf  # fallback to last try

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(best_buf.getvalue())


def find_eyes_landmarks(
    rgb: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    # Use Tasks API FaceLandmarker instead of deprecated solutions API
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "models",
        "mp_models",
        "face_landmarker",
        "face_landmarker.task",
    )
    with open(model_path, "rb") as f:
        model = f.read()

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_buffer=model),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,
        min_face_detection_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        img_h, img_w = rgb.shape[0], rgb.shape[1]
        candidates = []

        # Indices consistent with user's prior code
        idx_left, idx_right = 130, 359

        for face_lm in result.face_landmarks:
            try:
                le = face_lm[idx_left]
                re = face_lm[idx_right]
            except IndexError:
                continue
            left = (le.x * img_w, le.y * img_h)
            right = (re.x * img_w, re.y * img_h)
            dist = np.hypot(right[0] - left[0], right[1] - left[1])
            candidates.append((dist, left, right))

        if not candidates:
            return None

        # Pick the largest (closest face)
        _, left_pt, right_pt = max(candidates, key=lambda t: t[0])
        return left_pt, right_pt


def rotate_image_keep_center(
    bgr: np.ndarray, center: Tuple[float, float], angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, M


def apply_affine_to_point(
    M: np.ndarray, pt: Tuple[float, float]
) -> Tuple[float, float]:
    x, y = pt
    x_r = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    y_r = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return x_r, y_r


def scale_image(bgr: np.ndarray, scale: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def pad_to_fit(
    bgr: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[np.ndarray, int, int]:
    ih, iw = bgr.shape[:2]
    left = max(0, -x)
    top = max(0, -y)
    right = max(0, x + w - iw)
    bottom = max(0, y + h - ih)

    if any(v > 0 for v in (left, top, right, bottom)):
        bgr = cv2.copyMakeBorder(
            bgr, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE
        )
        x += left
        y += top
    return bgr, x, y


def crop_final(
    bgr: np.ndarray, spec: StandardSpec, eye_center_scaled: Tuple[float, float]
) -> np.ndarray:
    final_w, final_h = spec.width, spec.height
    desired_eye_y_from_top = final_h - (spec.eye_y_from_bottom_pct / 100.0) * final_h
    desired_eye_x = (spec.horiz_center_pct / 100.0) * final_w

    # Determine crop top-left so that eye_center lands at (desired_eye_x, desired_eye_y_from_top)
    tl_x = int(round(eye_center_scaled[0] - desired_eye_x))
    tl_y = int(round(eye_center_scaled[1] - desired_eye_y_from_top))

    # Ensure area exists (replicate-pad if needed)
    bgr, tl_x, tl_y = pad_to_fit(bgr, tl_x, tl_y, final_w, final_h)
    roi = bgr[tl_y : tl_y + final_h, tl_x : tl_x + final_w]

    # If numerical rounding produced mismatched size, fix it
    if roi.shape[0] != final_h or roi.shape[1] != final_w:
        roi = cv2.resize(roi, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
    return roi


def process_image_to_standard(
    inp_path: Path, spec: StandardSpec, max_tilt_deg: float
) -> np.ndarray:
    # Load
    bgr = load_image_bgr(inp_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Detect eyes
    eyes = find_eyes_landmarks(rgb)
    if eyes is None:
        raise ValueError(
            "No face/eyes detected. Ensure the image contains a clear, single face."
        )

    left, right = eyes
    # Angle (positive means right eye lower than left in image coords)
    dy = right[1] - left[1]
    dx = right[0] - left[0]
    angle_deg = np.degrees(np.arctan2(dy, dx))
    if abs(angle_deg) > max_tilt_deg:
        print(
            f"Warning: large initial eye tilt ({angle_deg:.1f}°). Proceeding with correction.",
            file=sys.stderr,
        )

    # Rotate around eye midpoint to make eyes horizontal
    eye_center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    rotated, M = rotate_image_keep_center(bgr, eye_center, -angle_deg)

    # Transform eye points to rotated space
    left_r = apply_affine_to_point(M, left)
    right_r = apply_affine_to_point(M, right)
    eye_center_r = (
        (left_r[0] + right_r[0]) / 2.0,
        (left_r[1] + right_r[1]) / 2.0,
    )

    # Determine scale so that pupil distance matches spec
    current_eye_dist = np.hypot(right_r[0] - left_r[0], right_r[1] - left_r[1])
    desired_eye_dist_px = (spec.pupil_dist_pct_of_width / 100.0) * spec.width
    if current_eye_dist < 1e-6:
        raise ValueError("Detected eye distance is too small to scale reliably.")
    scale = desired_eye_dist_px / current_eye_dist

    scaled = scale_image(rotated, scale)
    eye_center_scaled = (eye_center_r[0] * scale, eye_center_r[1] * scale)

    # Crop to final canvas with desired eye placement
    final_bgr = crop_final(scaled, spec, eye_center_scaled)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return final_rgb


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.input:
        parser.print_help(sys.stderr)
        sys.exit(2)

    inp = Path(args.input)
    if not inp.exists() or not inp.is_file():
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    # Build standard from preset + overrides
    spec = PRESETS[args.preset]
    spec = StandardSpec(
        width=args.width if args.width else spec.width,
        height=args.height if args.height else spec.height,
        eye_y_from_bottom_pct=args.eyes_y_bottom
        if args.eyes_y_bottom is not None
        else spec.eye_y_from_bottom_pct,
        pupil_dist_pct_of_width=args.pupil_dist
        if args.pupil_dist is not None
        else spec.pupil_dist_pct_of_width,
        horiz_center_pct=args.h_center
        if args.h_center is not None
        else spec.horiz_center_pct,
    ).clamp()

    out_path = resolve_output_path(inp, args)
    if out_path.exists() and not args.force:
        new_path = prompt_overwrite(out_path)
        if new_path is None:
            sys.exit(0)
        out_path = new_path

    try:
        final_rgb = process_image_to_standard(inp, spec, max_tilt_deg=args.max_tilt_deg)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        save_jpeg_with_target_size(
            final_rgb,
            out_path,
            target_kb=args.target_size_kb,
            tol_kb=args.size_tolerance_kb,
        )
    except Exception as e:
        print(f"Failed to save JPEG: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
