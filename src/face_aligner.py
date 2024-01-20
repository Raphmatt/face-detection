import cv2
import numpy as np


class FaceAligner:
    def __init__(self,
                 eye_spacing=(0.36, 0.40),
                 desired_width=1024,
                 desired_height=1280):
        """
        Aligns a face to a desired size and eye spacing
        :param eye_spacing: The space of the eyes and the edge of the image
        :param desired_width: The desired width of the aligned face
        :param desired_height: The desired height of the aligned face
        """
        # store the desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = eye_spacing
        self.desiredWidth = desired_width
        self.desiredHeight = desired_height

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredHeight is None:
            self.desiredHeight = self.desiredWidth

    def align(self, image: np.ndarray, left_eye_pts, right_eye_pts):
        # compute the angle between the eye centroids
        d_y = right_eye_pts[1] - left_eye_pts[1]
        d_x = right_eye_pts[0] - left_eye_pts[0]
        angle = np.degrees(np.arctan2(d_y, d_x))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desired_dist = (desired_right_eye_x - self.desiredLeftEye[0])
        desired_dist *= self.desiredWidth
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_pts[0] + right_eye_pts[0]) // 2,
                      (left_eye_pts[1] + right_eye_pts[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        material = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = self.desiredWidth * 0.5
        t_y = self.desiredHeight * self.desiredLeftEye[1]
        material[0, 2] += (t_x - eyes_center[0])
        material[1, 2] += (t_y - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desiredWidth, self.desiredHeight)
        output = cv2.warpAffine(image, material, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output
