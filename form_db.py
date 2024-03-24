import json
import os
from person import Person


class FormingDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        pass

    def write_db(self, persons):
        # делает db.json из массива persons
        all_face_encodings = [{'persons': []}]
        for person in persons:
            all_face_encodings[0]['persons'].append(person.to_json())

        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(all_face_encodings, f, ensure_ascii=False)

    def read_db(self):
        # делает массив persons из db.json
        persons = list()
        if os.path.isfile(self.db_path):
            with (open(self.db_path, "r", encoding='utf-8')) as openfile:
                all_face_encodings = json.load(openfile)
                for i in range(len(all_face_encodings[0]['persons'])):
                    person = Person()
                    person.id = all_face_encodings[0]['persons'][i]['id']
                    person.fullname = all_face_encodings[0]['persons'][i]['fullname']
                    person.encoding = all_face_encodings[0]['persons'][i]['encoding']
                    persons.append(person)
        return persons
