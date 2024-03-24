import cv2
import dlib
import numpy as np

from face_detection.dlib_detector import DlibDetector
from face_detection.retina_face_detector import RetinaFaceDetector
from face_encoding.encoder import Encoder


class DlibEncoder(Encoder):
    def __init__(self):
        self.face_detector = RetinaFaceDetector()
        self.encoding_predictor = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")

    def encode(self, img: np.ndarray, is_test=False):
        """
        Метод находит на изображении лицо, получает биометрические точки и преобразует их в вектор.
        :param is_test: при значении True созраняет изображения с рамкой и ключевыми точками
        :param img: изображение с камеры
        :return: вектор биометрии
        """
        # обнаружение лица на фото
        try:
            face = self.face_detector.detect_and_align(img, True)
        except Exception:
            return None

        # построение вектора биометрии
        img_representation = self.encoding_predictor.compute_face_descriptor(face)
        img_representation = np.array(img_representation)

        if is_test:
            self.save_img_with_rectangle_and_points(face, img)

        return img_representation


    def save_img_with_rectangle_and_points(self, faces, img):
        """
        Метод сохраняет изображение с рамкой и изображение с точками в /test_results

        """
        rectangle: dlib.rectangle = faces[0]
        a, b = rectangle.tl_corner(), rectangle.br_corner()
        # рисование рамки
        image = cv2.rectangle(img, (a.x, a.y), (b.x, b.y), color=(255, 0, 0), thickness=2)

        # сохранение картинки с рамкой
        cv2.imwrite("test_results/frame.jpg", image)
