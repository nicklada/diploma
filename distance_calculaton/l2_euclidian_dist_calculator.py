import numpy as np
from distance_calculaton.euclidian_dist_calculator import EuclidianCalculator


class EuclidianL2Calculator(EuclidianCalculator):

    def l2_normalize(self, x: np.ndarray) -> np.ndarray:
        return x / np.sqrt(np.sum(np.multiply(x, x)))

    def calculate(self, vector_a: np.ndarray, vector_b: np.ndarray):
        return super().calculate(self.l2_normalize(vector_a), self.l2_normalize(vector_b))
