import cv2
import dlib
import numpy as np
from retinaface import RetinaFace
from face_detection.detector import Detector


class RetinaFaceDlibDetector(Detector):
    def __init__(self):
        self.dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")
        self.retina_face_model = RetinaFace.build_model()

    def detect(self, img: np.ndarray, is_test=False):
        # обнаружение лица на фото и построение рамки
        faces = RetinaFace.detect_faces(img, model=self.retina_face_model)
        if 'face_1' not in faces:
            return None

        facial_area = faces['face_1']["facial_area"]
        rectangle = dlib.rectangle(facial_area[0], facial_area[1], facial_area[2], facial_area[3])

        # построение точек и вырезание лица по точкам
        img_shape: dlib.full_object_detection = self.dlib_facelandmark(img, rectangle)
        img_extracted = self.extract_face(img, img_shape)
        # выравнивание изображения
        img_aligned = dlib.get_face_chip(img_extracted, img_shape)
        if is_test:
            self.save_detected_and_aligned_face(img_aligned)

        return img_aligned

    def save_detected_and_aligned_face(self, face):
        """
        Метод сохраняет детектированное и выравненное лицо в /test_results

        """
        # сохранение обработанного развернутого изображения
        cv2.imwrite(f'test_results/debug/img_aligned{self.cnt}.jpg', face)

    #
    # def save_img_with_rectangle_and_points(self, face, img, img_shape):
    #     """
    #     Метод сохраняет изображение с рамкой и изображение с точками в /test_results
    #
    #     """
    #     rectangle: dlib.rectangle = face
    #     a, b = rectangle.tl_corner(), rectangle.br_corner()
    #     # рисование рамки
    #     image = cv2.rectangle(img, (a.x, a.y), (b.x, b.y), color=(255, 0, 0), thickness=2)
    #
    #     for point in img_shape.parts():
    #         image = cv2.circle(image, (point.x, point.y), radius=2, color=(0, 0, 255), thickness=2)
    #
    #     # сохранение картинки с рамкой
    #     cv2.imwrite("test_results/frame.jpg", image)

    def extract_face(self, img, img_shape):
        landmark_tuple = []

        for i in range(0, 68):
            x = img_shape.part(i).x
            y = img_shape.part(i).y
            landmark_tuple.append((x, y))

        routes = []

        for i in range(15, -1, -1):
            from_coordinate = landmark_tuple[i + 1]
            to_coordinate = landmark_tuple[i]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[0]
        to_coordinate = landmark_tuple[17]
        routes.append(from_coordinate)

        for i in range(17, 20):
            from_coordinate = landmark_tuple[i]
            to_coordinate = landmark_tuple[i + 1]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[19]
        to_coordinate = landmark_tuple[24]
        routes.append(from_coordinate)

        for i in range(24, 26):
            from_coordinate = landmark_tuple[i]
            to_coordinate = landmark_tuple[i + 1]
            routes.append(from_coordinate)

        from_coordinate = landmark_tuple[26]
        to_coordinate = landmark_tuple[16]
        routes.append(from_coordinate)
        routes.append(to_coordinate)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(bool)

        out = np.zeros_like(img)
        out[mask] = img[mask]

        return out
