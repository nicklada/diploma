import cv2
import dlib
import numpy as np

from face_encoding.encoder import Encoder


class DlibEncoder(Encoder):
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.encoding_predictor = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")

    def encode(self, img: np.ndarray, is_test=False) -> np.ndarray:
        """
        Метод находит на изображении лицо, получает биометрические точки и преобразует их в вектор.
        :param is_test: при значении True созраняет изображения с рамкой и ключевыми точками
        :param img: изображение с камеры
        :return: вектор биометрии
        """
        # обнаружение лица на фото
        faces = self.face_detector(img)

        if len(faces) == 1:
            # получение 68 ключевых точек
            img_shape: dlib.full_object_detection = self.dlib_facelandmark(img, faces[0])
            # выравнивание изображения
            img_aligned = dlib.get_face_chip(img, img_shape)
            # построение вектора биометрии
            img_representation = self.encoding_predictor.compute_face_descriptor(img_aligned)
            img_representation = np.array(img_representation)

            if is_test:
                self.save_img_with_rectangle_and_points(faces, img, img_shape)

            return img_representation
        else:
            return None

    def save_img_with_rectangle_and_points(self, faces, img, img_shape):
        """
        Метод сохраняет изображение с рамкой и изображение с точками в /test_results

        """
        rectangle: dlib.rectangle = faces[0]
        a, b = rectangle.tl_corner(), rectangle.br_corner()
        # рисование рамки
        image = cv2.rectangle(img, (a.x, a.y), (b.x, b.y), color=(255, 0, 0), thickness=2)

        # сохранение картинки с рамкой
        cv2.imwrite("test_results/frame.jpg", image)
        for point in img_shape.parts():
            image = cv2.circle(image, (point.x, point.y), radius=2, color=(0, 0, 255), thickness=2)
        cv2.imwrite("test_results/points.jpg", image)
