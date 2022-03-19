import numpy as np
from conv_mf.algorithm import StandardGradDescent, VectorizedLossFunctionMixin
from conv_mf.algorithm.func import sigmoid


class VectorizedLogisticRegression(VectorizedLossFunctionMixin):

    def __init__(self, learning_rate: float, itr: int, loss_func_name: str, l2_reg_param: float, output_dim: int):
        self._itr = itr
        self._loss_func = self.select_loss_func(loss_func_name, l2_reg_param)
        self._grad_descent = StandardGradDescent(learning_rate)
        self._output_dim = output_dim
        self._weights = None
        self._loss_history = []

    def _init_weights(self, dim: int) -> np.ndarray:
        return 0.1 * np.random.randn(dim, self._output_dim)

    def train(self, Xs: np.ndarray, ys: np.ndarray):
        # 行列計算をするために もし ys の shape が (N, ) であったら (N, 1) に変える(N はデータ数)
        if len(ys.shape) == 1:
            ys = np.expand_dims(ys, 1)

        data_dim = Xs.shape[1]
        self._weights = self._init_weights(data_dim)

        for _ in range(self._itr):
            pred_ys = self.predict_and_update_weights(Xs, ys)
            self._loss_history.append(
                self._loss_func.value(ys, pred_ys, self._weights)
            )
        return self

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
        if self._weights is None:
            raise ValueError('this model has not been trained.')
        return sigmoid(np.dot(Xs, self._weights))

    def get_loss_history(self):
        return self._loss_history
