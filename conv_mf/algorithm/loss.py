import numpy as np


class LossFunctionSelectMixin:

    def select_loss_func(self, name: str, l2_reg_param: float):
        if name == 'cross_entropy':
            return BinaryCrossEntropy(l2_reg_param)
        elif name == 'mean_square_error' or name == 'MSE':
            return MeanSquareError(l2_reg_param)
        else:
            raise ValueError('unsupported loss function: {0}'.format(name))


class VectorizedLossFunctionMixin:

    def select_loss_func(self, name: str, l2_reg_param: float):
        if name == 'cross_entropy':
            return VectorizedBinaryCrossEntropy(l2_reg_param)
        elif name == 'mean_square_error' or name == 'MSE':
            return VectorizedMeanSquareError(l2_reg_param)
        else:
            raise ValueError('unsupported loss function: {0}'.format(name))


class BinaryCrossEntropy:
    # log に 0 が入ると発散するので、小さい値を入れておく
    # DELTA を入れることで 1.0 log (0.0) のような場合をマイナスの大きい値として扱う
    DELTA = 1e-7

    def __init__(self, l2_reg_param: float = 0.0):
        self._l2_reg_param = l2_reg_param

    def grad(self, Xs: np.ndarray, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(Xs) != len(ys) or len(Xs) != len(pred_ys):
            raise ValueError(
                'ys and y must be the same length of Xs. len(Xs): {0} len(ys): {1}, len(pred_ys): {2}'.format(len(Xs), len(ys), len(pred_ys)))
        return np.dot(Xs.T, pred_ys - ys) / len(Xs) + 2.0 * self._l2_reg_param * weights

    def value(self, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> float:
        if len(ys) != len(pred_ys):
            raise ValueError(
                'ys must be the same length of pred_ys. len(ys): {0}, len(pred_ys): {1}'.format(len(ys), len(pred_ys)))
        num_data = len(ys)
        return - sum([
            y * np.log(pred_y + self.DELTA) + (1.0 - y) *
            np.log(1.0 - pred_y + self.DELTA)
            for y, pred_y in zip(ys[:, 0], pred_ys[:, 0])
            # weifghtsの掛け算をしたあと [0][0] 要素を取り出しているのは (M, 1) と (M, 1) をかけて (1, 1) ができるので要素を取り出すため
        ]) / num_data + self._l2_reg_param * np.dot(weights.T, weights)[0][0]


class VectorizedBinaryCrossEntropy:
    # log に 0 が入ると発散するので、小さい値を入れておく
    # DELTA を入れることで 1.0 log (0.0) のような場合をマイナスの大きい値として扱う
    DELTA = 1e-7

    def __init__(self, l2_reg_param: float = 0.0):
        self._l2_reg_param = l2_reg_param

    def grad(self, Xs: np.ndarray, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(Xs) != len(ys) or len(Xs) != len(pred_ys):
            raise ValueError(
                'ys and y must be the same length of Xs. len(Xs): {0} len(ys): {1}, len(pred_ys): {2}'.format(len(Xs), len(ys), len(pred_ys)))
        return np.dot(Xs.T, pred_ys - ys) / len(Xs) + 2.0 * self._l2_reg_param * weights

    def value(self, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> float:
        if len(ys) != len(pred_ys):
            raise ValueError(
                'ys must be the same length of pred_ys. len(ys): {0}, len(pred_ys): {1}'.format(len(ys), len(pred_ys)))

        num_data = len(pred_ys)
        loss = 0.0
        for dim_index in range(pred_ys.shape[1]):
            loss += - sum([
                y * np.log(pred_y + self.DELTA) + (1.0 - y) *
                np.log(1.0 - pred_y + self.DELTA)
                for y, pred_y in zip(ys[:, 0], pred_ys[:, dim_index])
                # weifghtsの掛け算をしたあと [0][0] 要素を取り出しているのは (M, 1) と (M, 1) をかけて (1, 1) ができるので要素を取り出すため
            ]) / num_data + self._l2_reg_param * np.dot(weights.T, weights)[dim_index][dim_index]

        return loss


class MeanSquareError:

    def __init__(self, l2_reg_param: float = 0.0):
        self._l2_reg_param = l2_reg_param

    def grad(self, Xs: np.ndarray, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(Xs) != len(ys) or len(Xs) != len(pred_ys):
            raise ValueError(
                'ys and y must be the same length of Xs. len(Xs): {0} len(ys): {1}, len(pred_ys): {2}'.format(len(Xs), len(ys), len(pred_ys)))
        return np.dot(Xs.T, (pred_ys - ys) * pred_ys * (1.0 - pred_ys)) / len(Xs) + 2.0 * self._l2_reg_param * weights

    def value(self, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> float:
        if len(ys) != len(pred_ys):
            raise ValueError(
                'ys must be the same length of pred_ys. len(ys): {0}, len(pred_ys): {1}'.format(len(ys), len(pred_ys)))
        num_data = len(ys)
        return 1.0 / 2.0 * sum([
            (y - pred_y) ** 2
            for y, pred_y in zip(ys[:, 0], pred_ys[:, 0])
        ]) / num_data + self._l2_reg_param * np.dot(weights.T, weights)[0][0]


class VectorizedMeanSquareError:

    def __init__(self, l2_reg_param: float = 0.0):
        self._l2_reg_param = l2_reg_param

    def grad(self, Xs: np.ndarray, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(Xs) != len(ys) or len(Xs) != len(pred_ys):
            raise ValueError(
                'ys and y must be the same length of Xs. len(Xs): {0} len(ys): {1}, len(pred_ys): {2}'.format(len(Xs), len(ys), len(pred_ys)))
        return np.dot(Xs.T, (pred_ys - ys) * pred_ys * (1.0 - pred_ys)) / len(Xs) + 2.0 * self._l2_reg_param * weights

    def value(self, ys: np.ndarray, pred_ys: np.ndarray, weights: np.ndarray) -> float:
        if len(ys) != len(pred_ys):
            raise ValueError(
                'ys must be the same length of pred_ys, len(ys): {0}, len(pred_ys): {1}'.format(
                    len(ys), len(pred_ys))
            )

        num_data = len(ys)
        loss = 0.0
        for dim_index in range(pred_ys.shape[1]):
            loss += 1.0 / 2.0 * sum([
                (y - pred_y) ** 2
                for y, pred_y in zip(ys[:, 0], pred_ys[:, dim_index])
            ]) / num_data + self._l2_reg_param * np.dot(weights.T, weights)[dim_index][dim_index]

        return loss
