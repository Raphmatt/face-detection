import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Function to detect occlusion based on visibility of landmarks
def detect_occlusion(face_landmarks):
    # Define indexes of landmarks for eyes and lips
    left_eye_indices = list(range(362, 382))
    right_eye_indices = list(range(133, 153))
    lips_indices = list(range(78, 95)) + list(range(308, 324))

    # Combine indices for easier processing
    combined_indices = left_eye_indices + right_eye_indices + lips_indices

    # Count how many landmarks are not visible
    not_visible_count = sum(1 for idx in combined_indices if face_landmarks.landmark[idx].visibility < 0.5)

    # Heuristic: if a significant number of landmarks are not visible, assume occlusion
    return not_visible_count > len(combined_indices) * 0.1  # Adjust the ratio as needed

# Initialize face mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect faces
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        occlusion_detected = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks for visualization
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Check for occlusion
                if detect_occlusion(face_landmarks):
                    occlusion_detected = True
                    break  # If occlusion is detected, no need to check further

        print("No Occlusion Detected" if occlusion_detected else "Occlusion Detected")

        # Display the image
        cv2.imshow('MediaPipe FaceMesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
