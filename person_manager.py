import os
import cv2
from face_detection.detector import Detector
from face_encoding.encoder import Encoder
from form_db import FormingDB
from person import Person


class PersonManager:
    def __init__(self, detector: Detector, encoder: Encoder, db_path: str, img_path: str = "data_20"):
        self.img_path = img_path
        self.db = FormingDB(db_path)
        self.detector: Detector = detector
        self.encoder: Encoder = encoder
        self.persons = self.db.read_db()
        pass

    # делает ДБ в формате json из данных в директории data_face
    def add_persons_to_db(self):
        """
        Метод парсит директорию data_face, получает id, фио и фото человека
        все данные сохраняются в массив persons
        :return:
        """
        for person_id in os.listdir(self.img_path):
            person_path = os.path.join(self.img_path, person_id)

            # проходит по self.persons: если Person.id == person_id то возвращает объект Person,
            # если в массиве self.persons не существует Person c Person.id == person_id - возвращает None
            person = next((x for x in self.persons if x.id == person_id), None)

            if person is None:
                # создаем объект Person c person_id и добавляем его в массив persons
                person = Person()
                person.id = person_id
                self.persons.append(person)

            # для каждого файла в директории person_path - если формат txt, считываем строку и заполняем person.fullname
            # если формат jpg or png, считываем изображение и заполняем person.img
            for file in os.listdir(person_path):
                file_path = os.path.join(person_path, file)

                if file[-3:] == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        person.fullname = f.readline()
                elif file[-3:] == 'jpg' or "png":
                    person.img = cv2.imread(file_path)
            if person.fullname == '':
                person.fullname = person_id

            # если изображение у объекта person существует - построить по нему вектор биометрии
            if person.img is not None:
                detected_img = self.detector.detect(person.img)
                if detected_img is None:
                    print("Не удалось обнаружить лицо ", person.id)
                    self.persons.remove(person)
                    continue
                encoding = self.encoder.encode(detected_img)
                person.encoding = encoding
                if encoding is None:
                    print("Не удалось построить вектор биометрии для ", person.id)
                    self.persons.remove(person)
            # если изображение у объекта person не существует - не добавлять человека в бд и вывести сообщение
            if person.img is None:
                print("В директории пользователя отсутствует изображение для ", person.id)
                self.persons.remove(person)

            print("Обработан: ", person.id)

        # сохранить итоговый массив persons в БД
        self.db.write_db(self.persons)
