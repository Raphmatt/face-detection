import cv2
import mediapipe as mp


async def recognize_face(image):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_image)

    num_faces = 0
    if results.detections:
        num_faces = len(results.detections)
        print(f"Number of faces found: {num_faces}")

        if num_faces == 1:
            bboxc = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = rgb_image.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
            print(bbox)
            x, y, w, h = bbox

            x_crop = max(0, x - 300)
            y_crop = max(0, y - 300)
            w_crop = min(width - x_crop, w + 600)
            h_crop = min(height - y_crop, h + 600)

            cropped_image = image[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]

        else:
            print("Found more than one face")
    else:
        print("No faces found in the image.")


async def process_face(image):
    # Implement face processing logic
    pass
