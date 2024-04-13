import numpy as np

from face_detection.detector import Detector


class DummyFaceDetector(Detector):
    def detect(self, img: np.ndarray, is_test=False):
        return img

