from distance_calculaton.distance_calculator import Calculator
from face_encoding.encoder import Encoder
from person_manager import PersonManager
import numpy as np


class Authenticator:
    def __init__(self, encoder, calculator, db_path):
        self.person_manager = PersonManager(encoder, db_path)
        self.encoder: Encoder = encoder
        self.calculator: Calculator = calculator

    def authenticate(self, img):
        """
        Метод запускает алгоритм распознавания лиц
        :return:
        """

        encoding = self.encoder.encode(img)

        if encoding is not None:
            is_auth = False
            for person in self.person_manager.persons:
                res = self.calculator.calculate(encoding, np.array(person.encoding))
                if res < 0.9:
                    print(f'it is {person.fullname}')
                    print(res)
                    is_auth = True

            if not is_auth:
                print('Лицо не найдено в базе')
        else:
            print('Лицо на изображении не обнаружено')
