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

import sys
import os
import argparse
import time
import cv2
import numpy as np
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import mediapipe as mp
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Make sure you're running this from the face-detection project root directory"
    )
    sys.exit(1)


async def process_frame_simple_alignment(frame: np.ndarray) -> Optional[np.ndarray]:
    """Simple face alignment - just straighten and center the face"""
    try:
        from face_utilities import get_face_details
        from face_aligner import FaceAligner

        # Get face details using MediaPipe only
        face_angle, left, right = get_face_details(frame, method="mediapipe")

        if face_angle is None or left is None or right is None:
            return None  # No face detected

        # Simple face alignment without validation or background removal
        eye_spacing = (0.36, 0.4)  # spacing_side/2, spacing_top
        aligner = FaceAligner(
            eye_spacing=eye_spacing, desired_width=512, desired_height=640
        )

        # Just align the face - no background processing
        aligned_image, out_of_bounds, rgba_aligned_image = aligner.align(
            frame, left, right
        )

        return aligned_image

    except Exception as e:
        print(f"Processing error: {e}")
        return None


def draw_info_overlay(
    frame: np.ndarray, fps: float, error_msg: str = None
) -> np.ndarray:
    """Draw FPS and status information on the frame"""
    overlay = frame.copy()

    # FPS counter
    cv2.putText(
        overlay,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Error message if any
    if error_msg:
        cv2.putText(
            overlay,
            f"Error: {error_msg}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    # Instructions
    cv2.putText(
        overlay,
        "ESC: Exit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return overlay


async def main():
    parser = argparse.ArgumentParser(description="Real-time face correction demo")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Camera width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Camera height (default: 480)"
    )

    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Using camera {args.camera}")
    print("Press ESC to exit")

    # FPS tracking
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Convert BGR to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame - simple alignment only
        processed_frame = await process_frame_simple_alignment(rgb_frame)

        # Create display layout
        display_frame = frame.copy()
        error_msg = None

        if processed_frame is not None:
            # Convert processed frame back to BGR and resize to match original
            if processed_frame.shape[2] == 4:  # RGBA
                processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)
            else:
                processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            # Resize processed frame to match original frame height
            h_orig = frame.shape[0]
            w_processed = int(processed_bgr.shape[1] * h_orig / processed_bgr.shape[0])
            processed_resized = cv2.resize(processed_bgr, (w_processed, h_orig))

            # Create side-by-side display
            if w_processed < frame.shape[1]:
                # Pad processed frame if it's narrower
                pad_width = frame.shape[1] - w_processed
                processed_padded = cv2.copyMakeBorder(
                    processed_resized,
                    0,
                    0,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                display_frame = np.hstack([frame, processed_padded])
            else:
                # Crop or resize if processed frame is wider
                processed_resized = cv2.resize(processed_bgr, (frame.shape[1], h_orig))
                display_frame = np.hstack([frame, processed_resized])
        else:
            error_msg = "No valid face detected"
            # Show only original frame with error
            display_frame = frame

        # Add overlay information
        display_frame = draw_info_overlay(display_frame, fps, error_msg)

        # Display the frame
        cv2.imshow("Face Correction Demo - Original | Processed", display_frame)

        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
