import dlib
import numpy as np

from face_encoding.encoder import Encoder


class DlibEncoder(Encoder):
    def __init__(self):
        self.encoding_predictor = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    def encode(self, img: np.ndarray, is_test=False):
        # построение вектора биометрии
        img_representation = self.encoding_predictor.compute_face_descriptor(img)
        img_representation = np.array(img_representation)

        return img_representation
