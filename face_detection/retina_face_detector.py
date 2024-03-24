import cv2
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from retinaface.commons import preprocess


class RetinaFaceDetector:

    def detect_and_align(self, img: np.ndarray, is_test=False):
        faces = RetinaFace.extract_faces(img_path=img, align=True)
        if len(faces) == 0:
            raise Exception("No faces found")
        face = cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB)
        face_sized = cv2.resize(face, [150, 150])

        if is_test:
            self.save_detected_and_aligned_face(face_sized)
        #face_sized = np.expand_dims(face_sized, axis=0)
        return face_sized

    def save_detected_and_aligned_face(self, face):
        """
        Метод сохраняет детектированное и выравненное лицо в /test_results

        """
        # сохранение обработанного развернутого изображения
        cv2.imwrite("test_results/detected_face.jpg", face)
