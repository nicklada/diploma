from distance_calculaton.distance_calculator import Calculator
from face_detection.detector import Detector
from face_encoding.encoder import Encoder
from person_manager import PersonManager
import numpy as np


class Authenticator:
    def __init__(self, detector, encoder, calculator, db_path, threshold):
        self.person_manager = PersonManager(detector, encoder, db_path)
        self.detector: Detector = detector
        self.encoder: Encoder = encoder
        self.calculator: Calculator = calculator
        self.threshold: float = threshold

    def authenticate(self, img):
        """
        Метод запускает алгоритм распознавания лиц
        :return:
        """
        detected_img = self.detector.detect(img)
        if detected_img is None:
            print("Не удалось обнаружить лицо ")
            return
        encoding = self.encoder.encode(detected_img)

        if encoding is not None:
            first_person = self.person_manager.persons[0]
            min_res = self.calculator.calculate(encoding, np.array(first_person.encoding))
            min_person_fullname = first_person.fullname

            for person in self.person_manager.persons:
                res = self.calculator.calculate(encoding, np.array(person.encoding))
                if res < min_res:
                    min_res = res
                    min_person_fullname = person.fullname

            if min_res < self.threshold:
                print(f'На фото: {min_person_fullname}. Расстояние - {min_res}')
            else:
                print('Лицо не найдено в базе')
        else:
            print('Лицо на изображении не обнаружено')
