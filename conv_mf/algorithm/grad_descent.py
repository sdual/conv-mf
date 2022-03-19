import numpy as np


class StandardGradDescent:

    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate

    def update_func(self, loss_grads: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return weights - self._learning_rate * loss_grads
