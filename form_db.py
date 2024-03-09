import json
import os

import cv2

from people_data import PeopleData


class FormingDB:
    def __init__(self, img_path, db_path):
        self.img_path = img_path
        self.db_path = db_path
        self.db = list()
        pass

    def write_db(self):
        all_face_encodings = [{'drivers': []}]
        for person in self.db:
            all_face_encodings[0]['drivers'].append(person.to_json())

        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(all_face_encodings, f, ensure_ascii=False)

    def read_db(self):
        if os.path.isfile(self.db_path):
            with (open(self.db_path, "r", encoding='utf-8')) as openfile:
                all_face_encodings = json.load(openfile)
                for i in range(len(all_face_encodings[0]['drivers'])):
                    id = all_face_encodings[0]['drivers'][i]['id']
                    encoding = all_face_encodings[0]['drivers'][i]['encoding']
                    fullname = all_face_encodings[0]['drivers'][i]['fullname']

                    people = PeopleData()
                    people.id = id
                    people.fullname = fullname
                    people.encoding = encoding
                    self.db.append(people)

    def create_img_data_list(self):
        """
        Метод парсит директорию data_face, получает id, фио и фото человека
        все данные сохраняются в объект PeopleData
        :return:
        """
        for face_id in os.listdir(self.img_path):
            person_path = os.path.join(self.img_path, face_id)

            person_data = next((x for x in self.db if x.id == face_id), None)
            if person_data is None:
                person_data = PeopleData()
                person_data.id = face_id
                self.db.append(person_data)

            for face in os.listdir(person_path):
                face_path = os.path.join(person_path, face)

                if face[-3:] == 'txt':
                    with open(face_path, 'r', encoding='utf-8') as f:
                        person_name = f.readline()
                        person_data.fullname = person_name
                elif face[-3:] == 'jpg' or "png":
                    img = cv2.imread(face_path)
                    person_data.img = img

