import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (0, 255, 196)
cap = cv2.VideoCapture(0)
prevTime = 0

with mp_selfie_segmentation.SelfieSegmentation(model_selection = 0) as selfie_segmentation:
    bg_image = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # For improved performance we pass image as not writable to pass by reference
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True

        # Bilateral Filter numpy
        condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1

        # Apply some background magic
        bg_image = cv2.imread('background/1.png')
        #bg_image = cv2.GaussianBlur(image, (55, 55), 0)


        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype = np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        cv2.putText(output_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 255, 2))


        cv2.imshow('DIY Background removal', output_image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
            break
