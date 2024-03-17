from encoder import Encoder
from person_manager import PersonManager
from utils import euclidean_dist


class Authenticator:
    def __init__(self):
        self.person_manager = PersonManager()
        self.encoder = Encoder()

    def authenticate(self, img):
        """
        Метод запускает алгоритм распознавания лиц
        :return:
        """
        encoding = self.encoder.encode(img)

        if encoding is not None:
            is_auth = False
            for person in self.person_manager.persons:
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
