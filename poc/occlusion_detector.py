import cv2
import dlib
import numpy as np

# Laden des Haar-Kaskaden-Klassifikators für die Gesichtserkennung
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Laden des Dlib Shape Predictors
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Pfad zum Dlib-Modell
predictor = dlib.shape_predictor(predictor_path)


# Funktion, um Gesichtsmerkmale zu erkennen
def detect_face_features(gray, rect):
    shape = predictor(gray, rect)
    shape = np.array([[p.x, p.y] for p in shape.parts()])
    return shape


# Funktion zur Okklusionsanalyse
def analyze_occlusions(landmarks):
    # Einfache Überprüfung, ob Augen und Mund innerhalb der Kieferlinie liegen
    jawline = landmarks[:17]
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    jawline_x = [p[0] for p in jawline]
    jawline_y = [p[1] for p in jawline]

    min_x, max_x = min(jawline_x), max(jawline_x)
    min_y, max_y = min(jawline_y), max(jawline_y)

    def is_within_jawline(feature):
        for x, y in feature:
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                return False
        return True

    if not all(
        [is_within_jawline(feature) for feature in [left_eye, right_eye, mouth]]
    ):
        print("Mögliche Okklusion erkannt.")
    else:
        print("Keine Okklusion erkannt.")


# Webcam erfassen
cap = cv2.VideoCapture(0)

while True:
    # Bild von der Webcam erfassen
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gesichter erkennen
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        face_rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
        landmarks = detect_face_features(gray, face_rect)

        # Okklusionsanalyse
        analyze_occlusions(landmarks)

        # Zeichnen der Gesichtsmerkmale zum Veranschaulichen
        for x, y in landmarks:
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    # Bild anzeigen
    cv2.imshow("Webcam", image)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Alles schließen, wenn die Schleife beendet wird
cap.release()
cv2.destroyAllWindows()
