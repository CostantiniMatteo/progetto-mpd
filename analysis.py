import smarthouse
import numpy as np
import itertools
import sklearn.metrics
import matplotlib.pyplot as plt

def predict(dataset, train_rate=None, days=None,
            n_samples=None, days_test=None):
    d = None
    to_date_test = None
    if days:
        start_A = smarthouse.date_to_timestamp("2011-11-28 00:00:00")
        start_B = smarthouse.date_to_timestamp("2012-11-11 00:00:00")
        d = {
            'A': start_A + 86400*(14 - days),
            'B': start_B + 86400*(21 - days)
        }
    if days_test:
        to_date_test = {
            'A': start_A + 86400*(14 - days + days_test),
            'B': start_B + 86400*(21 - days + days_test)
        }


    return smarthouse.main(
        datasets=[dataset],
        train_rate=train_rate,
        to_date=d,
        n_samples=n_samples,
        to_date_test=to_date_test
    )


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    # A - 3 giorni
    print("=== A - 3 giorni ===")
    truth_a, predict_a, accuracy_a = predict('A', days=3)
    print(sklearn.metrics.classification_report(truth_a, predict_a))
    conf_mat_a = sklearn.metrics.confusion_matrix(truth_a, predict_a)

    plt.figure(1)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(truth_a)))),
        normalize=True
    )

    # B - 4 giorni
    print("=== B - 4 giorni ===")
    truth_b, predict_b, accuracy_b = predict('B', days=4)
    print(sklearn.metrics.classification_report(truth_b, predict_b))
    conf_mat_b = sklearn.metrics.confusion_matrix(truth_b, predict_b)

    plt.figure(2)
    plot_confusion_matrix(
        conf_mat_b,
        list(map(str, range(max(truth_b)))),
        normalize=True
    )

    # A - 3000 samples
    print("=== A - 3000 samples ===")
    truth_a, predict_a, accuracy_a = predict('A', n_samples=3000)
    print(sklearn.metrics.classification_report(truth_a, predict_a))
    conf_mat_a = sklearn.metrics.confusion_matrix(truth_a, predict_a)

    plt.figure(3)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(truth_a)))),
        normalize=True
    )

    # A - 20000 samples
    print("=== A - 20000 samples ===")
    truth_a, predict_a, accuracy_a = predict('A', n_samples=20000)
    print(sklearn.metrics.classification_report(truth_a, predict_a))
    conf_mat_a = sklearn.metrics.confusion_matrix(truth_a, predict_a)

    plt.figure(4)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(truth_a)))),
        normalize=True
    )

    # B - 3000 samples
    print("=== B - 3000 samples ===")
    truth_b, predict_b, accuracy_b = predict('B', n_samples=3000)
    print(sklearn.metrics.classification_report(truth_b, predict_b))
    conf_mat_b = sklearn.metrics.confusion_matrix(truth_b, predict_b)

    plt.figure(5)
    plot_confusion_matrix(
        conf_mat_b,
        list(map(str, range(max(truth_b)))),
        normalize=True
    )

    # B - 20000 samples
    print("=== B - 20000 samples ===")
    truth_b, predict_b, accuracy_b = predict('B', n_samples=20000)
    print(sklearn.metrics.classification_report(truth_b, predict_b))
    conf_mat_b = sklearn.metrics.confusion_matrix(truth_b, predict_b)

    plt.figure(6)
    plot_confusion_matrix(
        conf_mat_b,
        list(map(str, range(max(truth_b)))),
        normalize=True
    )

    # A - Train su 4 giorni
    print("=== A - Train 4 giorni ===")
    truth_a, predict_a, accuracy_a = predict('A', days=10, days_test=3)
    print(sklearn.metrics.classification_report(truth_a, predict_a))
    conf_mat_a = sklearn.metrics.confusion_matrix(truth_a, predict_a)

    plt.figure(7)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(truth_a)))),
        normalize=True
    )

    print("=== A - Train 4 giorni ===")
    truth_a, predict_a, accuracy_a = predict('A', days=10, days_test=10)
    print(sklearn.metrics.classification_report(truth_a, predict_a))
    conf_mat_a = sklearn.metrics.confusion_matrix(truth_a, predict_a)

    plt.figure(8)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(truth_a)))),
        normalize=True
    )

    # B - Train su 4 giorni
    print("=== B - Train 4 giorni, Test 3 giorni ===")
    truth_b, predict_b, accuracy_b = predict('B', days=3)
    print(sklearn.metrics.classification_report(truth_b, predict_b))
    conf_mat_b = sklearn.metrics.confusion_matrix(truth_b, predict_b)

    plt.figure(9)
    plot_confusion_matrix(
        conf_mat_b,
        list(map(str, range(max(truth_b)))),
        normalize=True
    )

    print("=== B - Train 4 giorni, Test 17 giorni ===")
    truth_b, predict_b, accuracy_b = predict('B', days=17)
    print(sklearn.metrics.classification_report(truth_b, predict_b))
    conf_mat_b = sklearn.metrics.confusion_matrix(truth_b, predict_b)

    plt.figure(10)
    plot_confusion_matrix(
        conf_mat_b,
        list(map(str, range(max(truth_b)))),
        normalize=True
    )

    plt.show()
