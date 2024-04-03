import numpy as np
import dlib

from face_detection.detector import Detector


class DlibDetector(Detector):
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")

    def detect(self, img: np.ndarray, is_test=False):
        # обнаружение лица на фото
        faces = self.face_detector(img)
        if len(faces) == 0:
            raise Exception("No faces found")
        img_shape: dlib.full_object_detection = self.dlib_facelandmark(img, faces[0])
        # выравнивание изображения
        img_aligned = dlib.get_face_chip(img, img_shape)
        return img_aligned
