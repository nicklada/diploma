from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):
    @abstractmethod
    def encode(self, img: np.ndarray, is_test=False):
        pass
