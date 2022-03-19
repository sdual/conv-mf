import traceback

import matplotlib.pyplot as plt
from sklearn import datasets

from conv_mf.algorithm import LogisticRegression, VectorizedLogisticRegression


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :][:100]
    y = iris.target[:100]

    print('data loaded.')

    print(X.shape)

    # vlr = LogisticRegression(0.1, 200, 'MSE', 0.1)
    vlr = VectorizedLogisticRegression(0.1, 200, 'MSE', 0.1, 10)
    print('start to train.')
    vlr.train(X, y)

    print('loss history: ')
    loss_histry = vlr.get_loss_history()
    print(loss_histry)

    plt.plot(loss_histry)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
