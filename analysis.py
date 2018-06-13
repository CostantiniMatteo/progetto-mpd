import smarthouse
import numpy as np
import itertools
import sklearn.metrics
import matplotlib.pyplot as plt

def confusion_matrix(dataset, train_rate=None, days=None):
    if days:
        start_A = smarthouse.date_to_timestamp("2011-11-28 00:00:00")
        start_B = smarthouse.date_to_timestamp("2012-11-11 00:00:00")
        d = {
            'A': start_A + 86400*(14-days),
            'B': start_B + 86400*(21 - days)
        }

    truth, predict, accuracy = smarthouse.main(
        datasets=[dataset],
        train_rate=train_rate,
        to_date=d
    )

    conf_mat = sklearn.metrics.confusion_matrix(truth, predict)

    print(sklearn.metrics.classification_report(truth, predict))

    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(
        conf_mat,
        list(map(str, range(max(truth)))),
        normalize=True
    )
    plt.show()


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
    confusion_matrix('A', days=3)
