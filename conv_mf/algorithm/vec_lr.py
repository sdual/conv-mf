import numpy as np
from conv_mf.algorithm import BinaryCrossEntropy, StandardGradDescent


class VectorizedLogisticRegression:

    def __init__(self, learning_rate: float, itr: int, l2_reg_param: float, output_dim: int):
        self._itr = itr
        self._loss_func = BinaryCrossEntropy(l2_reg_param)
        self._grad_descent = StandardGradDescent(learning_rate)
        self._output_dim = output_dim
        self._weights = None
        self._loss_history = []

    @staticmethod
    def _init_weights(dim: int) -> np.ndarray:
        return 0.1 * np.random.randn(dim, 1)

    def train(self, Xs: np.ndarray, ys: np.ndarray):
        # ys は vector を予測するので二次元配列

        data_dim = Xs.shape[1]
        self._weights = self._init_weights(data_dim)

        for _ in range(self._itr):
            # pred_ys は vector を予測するので二次元配列
            pred_ys = self.predict_and_update_weights(Xs, ys)

    def predict_and_update_weights(self, Xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        # 全データの予測値を同時に計算
        pred_ys: np.ndarray = self.predict(Xs)
        loss_grads: np.ndarray = self._loss_func.grad(
            Xs, ys, pred_ys, self._weights)
        # parameter の更新
        self._weights = self._grad_descent.update_func(
            loss_grads, self._weights)
        return pred_ys

    def predict(self, Xs: np.ndarray) -> np.ndarray:
        pass

    def get_loss_history(self):
        return self._loss_history
