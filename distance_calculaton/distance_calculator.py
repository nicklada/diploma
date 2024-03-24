from abc import ABC, abstractmethod
import numpy as np


class Calculator(ABC):
    @abstractmethod
    def calculate(self, vector_a: np.ndarray, vector_b: np.ndarray) -> np.float64:
        pass
