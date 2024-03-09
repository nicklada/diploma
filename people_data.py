import json

from PIL import Image
from numpy import array, ndarray


class PeopleData:
    def __init__(self):
        self.id: str = ''
        self.fullname: str = ''
        self.encoding: array = []
        self.img: Image

    def to_json(self):
        res_dict: dict = dict()

        res_dict["id"] = self.id
        res_dict["fullname"] = self.fullname.replace('\n', '')
        if type(self.encoding) == ndarray:
            res_dict["encoding"] = self.encoding.tolist()
        else:
            res_dict["encoding"] = self.encoding

        return res_dict

    def __str__(self):
        return f'\n{self.id} \n{self.fullname}{self.img}'


class LandmarksInPeople:
    def __init__(self):
        self.landmarks = ""
        self.people_face = ""
        self.people = ""
