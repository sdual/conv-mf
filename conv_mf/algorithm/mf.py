import numpy as np


class ALS:

    def __init__(self, itr: int, alpha: float, beta: float, latent_dim: int):
        self._itr = itr
        self._alpha = alpha
        self._beta = beta
        self._latent_dim = latent_dim

    def train(self, r_matrix: np.ndarray, ys: np.ndarray):
        num_users = r_matrix.shape[0]
        num_items = r_matrix.shape[1]
        user_factors = np.random.randn(num_users, self._latent_dim)
        item_factors = np.random.randn(num_items, self._latent_dim)

        for itr_index in range(self._itr):
            for user_index in range(num_users):
                for item_index in range(num_items):
                    diff = r_matrix[user_index][item_index] - \
                        np.dot(user_factors[user_index, :],
                               item_factors[:, item_index])

        pass

    def _update_matrix_elemets(self, user_factors: np.ndarray, item_factors: np.ndarray):
        for user_index in range(num_users):
            for item_index in range(num_items):
                diff = r_matrix[user_index][item_index] - np.dot(user_factors[user_index, :],
                                                                 item_factors[:, item_index])
                grad = -2.0 * diff *

    def _update(self, value: float, grad: float) -> float:
        return value - self._alpha * grad

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        pass
