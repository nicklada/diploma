import numpy as np

from distance_calculaton.distance_calculator import Calculator


class EuclidianCalculator(Calculator):
    def __init__(self, normalization_coef=1):
        self.normalization_coef = normalization_coef

    def calculate(self, vector_a: np.ndarray, vector_b: np.ndarray):
        return np.sqrt(np.sum((vector_a - vector_b) ** 2)) / self.normalization_coef
