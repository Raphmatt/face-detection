import cv2
import dlib
import numpy as np

# Laden des Haar-Kaskaden-Klassifikators für die Gesichtserkennung
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
    # Symmetrieprüfung
    def symmetry_deviation(point_left, point_right, eye_center):
        return abs((point_left[0] - eye_center[0]) - (eye_center[0] - point_right[0]))

    eye_center = np.mean(landmarks[36:48], axis=0)
    sym_deviation = sum([symmetry_deviation(landmarks[i], landmarks[17-i], eye_center) for i in range(17)])
    sym_deviation += sum([symmetry_deviation(landmarks[i], landmarks[26-i], eye_center) for i in range(17, 22)])
    sym_deviation += sum([symmetry_deviation(landmarks[i], landmarks[48-i], eye_center) for i in range(48, 55)])
    avg_deviation = sym_deviation / (22 + 5 + 7)

    # Schwellenwert für Symmetrieabweichung festlegen
    symmetry_threshold = 10  # Beispielwert, sollte experimentell angepasst werden

    if avg_deviation > symmetry_threshold:
        print("Mögliche Okklusion durch Asymmetrie erkannt.")
    else:
        print("Keine Okklusion durch Asymmetrie erkannt.")

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
