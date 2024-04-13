from insightface.app import FaceAnalysis
import onnxruntime as ort
from face_encoding.encoder import Encoder
import numpy as np


class InsightFaceEncoder(Encoder):
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def encode(self, img: np.ndarray, is_test=False):
        emb_res = self.app.get(img)

        return emb_res[0].embedding
