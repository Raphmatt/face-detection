import cv2
import mediapipe as mp
import numpy as np

def detect_occlusion(image_np):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    result = face_detection.process(image_rgb)

    if result.detections:
        for detection in result.detections:
            if detection.location_data.relative_bounding_box.ymin < 0.2:
                return "Face covered by an object"
            else:
                return "Face not covered by an object"
    else:
        return "No face detected"

# Initialisiere die Kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Rufe die Methode auf
    message = detect_occlusion(frame)

    # Zeige das Bild und die Nachricht an
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Occlusion Detection', frame)

    # Beende die Schleife bei Betätigen der 'q'-Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigabe der Ressourcen
cap.release()
cv2.destroyAllWindows()
