import cv2
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from retinaface.commons import preprocess

from face_detection.detector import Detector


class RetinaFaceDetector(Detector):
    def __init__(self, size=None):
        if size is None:
            size = [112, 112]
        self.size = size

    def detect(self, img: np.ndarray, is_test=False):
        faces = RetinaFace.extract_faces(img_path=img, align=False)
        if len(faces) == 0:
            return None
        face = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
        face_sized = cv2.resize(face, self.size)

        if is_test:
            self.save_detected_and_aligned_face(face_sized)

        return face_sized

    def save_detected_and_aligned_face(self, face):
        """
        Метод сохраняет детектированное и выравненное лицо в /test_results

        """
        # сохранение обработанного развернутого изображения
        cv2.imwrite("test_results/detected_face.jpg", face)
