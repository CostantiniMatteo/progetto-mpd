from smarthouse import smarthouse
import numpy as np
import itertools
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


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


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    it = zip(pc.get_paths(), pc.get_facecolors(), pc.get_array())
    for p, color, value in it:
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel,
            xticklabels, yticklabels,
            figure_width=40, figure_height=20,
            correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed',
        linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report_2(truth, predict,
        title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    classification_report = sklearn.metrics.classification_report(truth, predict)
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
        for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels,
            yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


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
