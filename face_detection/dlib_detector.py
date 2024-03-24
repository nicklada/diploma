import numpy as np
import dlib


class DlibDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def detector(self, img: np.ndarray, is_test=False):
        # обнаружение лица на фото
        faces = self.face_detector(img)
        if is_test:
            self.save_img_with_rectangle_and_points(faces, img)
        return faces
