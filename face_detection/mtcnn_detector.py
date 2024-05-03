import numpy as np

from face_detection.detector import Detector
from facenet_pytorch import MTCNN
import torch


class MtcnnDetector(Detector):
    def __init__(self):
        self.face_detector = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

    def detect(self, img: np.ndarray, is_test=False):
        faces, _ = self.face_detector.detect(img)

        if faces is None:
            return None

        return self.face_detector(img)
