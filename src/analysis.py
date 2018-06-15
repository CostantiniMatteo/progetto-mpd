import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from smarthouse import smarthouse
from utils import plot_confusion_matrix


def predict(**kwargs):
    truth, predict, accuracy = smarthouse(**kwargs)
    print(sklearn.metrics.classification_report(truth, predict))
    conf_mat = sklearn.metrics.confusion_matrix(truth, predict)

    plot_confusion_matrix(
        conf_mat,
        list(map(str, range(max(truth)))),
        normalize=True
    )


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    # A - 3000 samples
    print("=== A - 3000 samples ===")
    plt.figure(1)
    predict(dataset='A', n_samples=3000)

    # A - 20000 samples
    print("=== A - 20000 samples ===")
    plt.figure(2)
    predict(dataset='A', n_samples=20000)

    # B - 3000 samples
    print("=== B - 3000 samples ===")
    plt.figure(3)
    predict(dataset='B', n_samples=3000)

    # B - 20000 samples
    print("=== B - 20000 samples ===")
    plt.figure(4)
    predict(dataset='B', n_samples=20000)

    # A - Train su 4 giorni
    print("=== A - Train 5 giorni, Test 3 giorni ===")
    plt.figure(5)
    predict(dataset='A', test_days=3)

    print("=== A - Train 5 giorni, Test 9 giorni ===")
    plt.figure(6)
    predict(dataset='A', test_days=9)

    # B - Train su 4 giorni
    print("=== B - Train 5 giorni, Test 3 giorni ===")
    plt.figure(7)
    predict(dataset='B', test_days=3)

    print("=== B - Train 5 giorni, Test 16 giorni ===")
    plt.figure(8)
    predict(dataset='B', test_days=16)

    plt.show()
