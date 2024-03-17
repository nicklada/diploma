import json
import os
import cv2
from authenticator import Authenticator
from person import Person
from person_manager import PersonManager

if __name__ == '__main__':
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    authenticator = Authenticator()
    test_img = cv2.imread('test_faces/m.0b309__0002.jpg')
    authenticator.authenticate(test_img)

# if __name__ == '__main__':
#     # создает объект класса PersonManager и вызывает метод add_persons_to_db,
#     # который делает ДБ в формате json из данных в директории data_face
#     person_manager = PersonManager()
#     person_manager.add_persons_to_db()


def start_cam(self):
    """
    Метод запускает алгоритм распознавания с использованием веб-камеры
    :return:
    """
    authenticator = Authenticator()
    cap = cv2.VideoCapture(0)

    frame = cap.read()
    authenticator.authenticate(frame[1])


def get_biometrical_point_and_write_to_file(img_path):
    a = Authenticator()

    i = 0
    full = len(os.listdir(img_path))
    asd = os.listdir(img_path)
    for face_id in os.listdir(img_path):
        print(f"\r ${i} из ${full}", end=' ')
        if face_id == ".DS_Store":
            continue

        person_path = os.path.join(img_path, face_id)
        all_face_encodings = [{'persons': []}]
        person_list = []

        for face in os.listdir(person_path):
            face_path = os.path.join(person_path, face)

            if face[-3:] == 'jpg' or face[-3:] == "png":
                img = cv2.imread(face_path)
                r = a.encoder.encode(img)
                person_data = Person()

                person_data.id = face_id
                person_data.fullname = face
                person_data.encoding = r
                person_list.append(person_data)

        for person in person_list:
            all_face_encodings[0]['persons'].append(person.to_json())

        face_path = os.path.join(person_path, "point.json")
        with open(face_path, 'w', encoding='utf-8') as f:
            json.dump(all_face_encodings, f, ensure_ascii=False)
        i += 1
