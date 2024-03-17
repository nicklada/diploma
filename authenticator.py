from form_db import FormingDB
import dlib
import cv2
import numpy as np
from utils import euclidean_dist


class Authenticator:
    def __init__(self, model_path):
        # создаем объект класса FormingDB
        self.form_db = FormingDB("data_face", "db_rus.json")
        self.form_db.read_db()
        self.landmarks = list()
        self.model_path = model_path
        self.face_detector = dlib.get_frontal_face_detector()
        self.encoding_predictor = dlib.face_recognition_model_v1(
            self.model_path + "dlib_face_recognition_resnet_model_v1.dat")
        self.dlib_facelandmark = dlib.shape_predictor(self.model_path + "shape_predictor_68_face_landmarks_GTX.dat")
        self.isTest = False

    def create_image_data_list(self):
        self.form_db.create_img_data_list()

    def get_encoding(self, img):
        """
        Метод находит на изобрадении лицо, получает биометрические точки
        и преобразует их в вектор.
        :param img: изображение с камеры
        :return: вектор биометрии
        """

        faces = self.face_detector(img)

        if len(faces) == 1:
            if self.isTest:
                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2

                rectangle: dlib.rectangle = faces[0]
                a, b = rectangle.tl_corner(), rectangle.br_corner()

                image = cv2.rectangle(img, (a.x, a.y), (b.x, b.y), color, thickness)

                # Displaying the image
                cv2.imwrite("test_results/frame.jpg", image)

            img_shape: dlib.full_object_detection = self.dlib_facelandmark(img, faces[0])
            img_aligned = dlib.get_face_chip(img, img_shape)
            img_representation = self.encoding_predictor.compute_face_descriptor(img_aligned)
            img_representation = np.array(img_representation)

            if self.isTest:
                for point in img_shape.parts():
                    image = cv2.circle(image, (point.x, point.y), radius=2, color=(0, 0, 255), thickness=2)
                cv2.imwrite("test_results/points.jpg", image)

            return img_representation
        else:
            return None

    def add_photo(self):
        """
        Метод обрабатывает имеющиеся фото в директории data_face
        и формирует на их основе БД лиц
        :return:
        """
        for person in self.form_db.db:
            encoding = self.get_encoding(person.img)
            person.encoding = encoding

        self.form_db.write_db()

    def start_cam(self):
        """
        Метод запускает алгоритм распознавания с использованием веб-камеры
        :return:
        """
        cap = cv2.VideoCapture(0)

        frame = cap.read()
        encoding = self.get_encoding(frame[1])

        if encoding is not None:
            is_auth = False
            for person in self.form_db.db:
                res = euclidean_dist(encoding, person.encoding)
                if res < 0.6:
                    print(f'it is {person.fullname}')
                    print(res)
                    is_auth = True
                    break
            if not is_auth:
                print('people not found')
        else:
            print('no person in camera')

    def run_pipeline(self, img):
        """
        Метод запускает алгоритм распознавания лиц
        :return:
        """

        encoding = self.get_encoding(img)

        if encoding is not None:
            is_auth = False
            for person in self.form_db.db:
                res = euclidean_dist(encoding, person.encoding)
                if res < 0.6:
                    print(f'it is {person.fullname}')
                    print(res)
                    is_auth = True
                    break
            if not is_auth:
                print('Лицо не найдено в базе')
        else:
            print('Лицо на изображении не обнаружено')
