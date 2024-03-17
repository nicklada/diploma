from PIL import Image
from numpy import array, ndarray


class Person:
    def __init__(self):
        self.id: str = ''
        self.fullname: str = ''
        self.encoding: array = []
        self.img: Image

    def to_json(self):
        return {
            "id": self.id,
            "fullname": self.fullname.replace('\n', ''),
            "encoding": self.encoding.tolist() if type(self.encoding) == ndarray else self.encoding
        }

    def __str__(self):
        return f'\n{self.id} \n{self.fullname}{self.img}'
