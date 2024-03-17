import json
import os
import cv2
from authenticator import Authenticator
from people_data import PeopleData

if __name__ == '__main__':
    # создает объект класса Authenticator, передает лицо из test_faces в пайплайн,
    # пайплайн строит вектор переданного лица и сравнивает его с векторами лиц из БД
    a = Authenticator('models/')
    a.isTest = True
    test_img = cv2.imread('test_faces/lada.jpg')
    a.run_pipeline(test_img)


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
