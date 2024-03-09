import json
import os

import cv2
import dlib
import numpy as np
from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi

from form_db import FormingDB
from people_data import PeopleData
from utils import set_brightness_img, saturation_img, euclidean_dist
from fdlite.render import Colors, landmarks_to_render_data, render_to_image, detections_to_render_data
from PIL import Image


class Authenticator:
    def __init__(self, model_path):
        #создаем объект класса FormingDB
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
                cv2.imwrite("window_name.jpg", image)

            # face_img_batch = [img,
            #                   cv2.flip(img.copy(), 0),
            #                   set_brightness_img(img.copy(), -60),
            #                   set_brightness_img(img.copy(), 40),
            #                   saturation_img(img.copy(), -60)]
            #
            # face_encodings = []
            # for face_img in face_img_batch:
            #     face_landmarks = dlib_facelandmark(face_img, faces[0])
            #     face_encoding = encoding_predictor.compute_face_descriptor(face_img, face_landmarks)
            #     face_encodings.append(face_encoding)
            #
            # nparray_face = np.array(face_encodings)
            # mean_face_encod = nparray_face.mean(axis=0)
            #
            # return mean_face_encod
            img_shape: dlib.full_object_detection = self.dlib_facelandmark(img, faces[0])
            img_aligned = dlib.get_face_chip(img, img_shape)
            img_representation = self.encoding_predictor.compute_face_descriptor(img_aligned)
            img_representation = np.array(img_representation)

            if self.isTest:
                for point in img_shape.parts():
                    image = cv2.circle(image, (point.x, point.y), radius=2, color=(0, 0, 255), thickness=2)
                cv2.imwrite("window_name2.jpg", image)

            return img_representation
        else:
            print('faces not found')
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
        while True:
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


if __name__ == '__main__':
    #создаем объект аутентификатора
    a = Authenticator('models/')
    a.create_image_data_list()
    a.isTest = True

    img = cv2.imread('data_face/5f6a41a68bb79b58ae2c729c/photo_2021-03-19_16-10-51.jpg')
    img1 = cv2.imread('test4.jpg')
    r = a.get_encoding(img)
    r1 = a.get_encoding(img1)
    res = euclidean_dist(r, r1)
    print(res)


# if __name__ == '__main__':
#     a = Authenticator('models/')
#     a.create_image_data_list()
#
# if __name__ == '__main__':
#     a = Authenticator('models/')
#     a.start_cam()

def get_biometrical_point_and_write_to_file(img_path):
    a = Authenticator('models/')

    i = 0
    full = len(os.listdir(img_path))
    asd = os.listdir(img_path)
    for face_id in os.listdir(img_path):
        print(f"\r ${i} из ${full}", end=' ')
        if face_id == ".DS_Store":
            continue

        person_path = os.path.join(img_path, face_id)
        all_face_encodings = [{'drivers': []}]
        person_list = []

        for face in os.listdir(person_path):
            face_path = os.path.join(person_path, face)

            if face[-3:] == 'jpg' or face[-3:] == "png":
                img = cv2.imread(face_path)
                r = a.get_encoding(img)
                person_data = PeopleData()

                person_data.id = face_id
                person_data.fullname = face
                person_data.encoding = r
                person_list.append(person_data)

        for person in person_list:
            all_face_encodings[0]['drivers'].append(person.to_json())

        face_path = os.path.join(person_path, "point.json")
        with open(face_path, 'w', encoding='utf-8') as f:
            json.dump(all_face_encodings, f, ensure_ascii=False)
        i += 1


# Запуск пайплайна:
# Чтение БД
if __name__ == '__main__':
    a = Authenticator('models/')
    a.create_image_data_list()
