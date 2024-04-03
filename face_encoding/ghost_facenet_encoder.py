import numpy as np

from face_encoding.encoder import Encoder
import tensorflow as tf
from sklearn.preprocessing import normalize


class GhostFaceNetEncoder(Encoder):
    def __init__(self):
        self.model = tf.keras.models.load_model("models/GhostFaceNet_W1.3_S1_ArcFace.h5", compile=False)

    def encode(self, img: np.ndarray, is_test=False):
        img_prepared = self.prepare_image(img)
        img_representation = self.model.predict(img_prepared)
        img_representation = normalize(np.array(img_representation).astype("float32"))[0]

        return img_representation

    def prepare_image(self, img):
        img = (img - 127.5) * 0.0078125
        img = np.expand_dims(img, axis=0)

        return img
