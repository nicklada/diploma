from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    @abstractmethod
    def detect(self, img: np.ndarray, is_test=False):
        pass
