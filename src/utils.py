from datetime import datetime
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import sys


# --- Varie ---
# Converte una data in formato YYYY-mm-dd HH:MM:SS in timestamp
def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


# Suddivide la giornata in 4 slice [0, 6) [6, 12) [12, 18) [18 24)
# Restituisce la frazione del giorno a cui appartiene il timestamp
def day_period(timestamp):
    h = (timestamp // (60 * 60)) % 24
    if h < 6:
        return 0
    elif h < 12:
        return 1
    elif h < 18:
        return 2
    else:
        return 3


def print_numpy_matrix(m):
    np.savetxt(sys.stdout, m, "%6.4f")


# --- Dataset ---
# Carica il dataset finale. name = {'A', 'B'}
# Se mapping=True viene restituito anche il dizionario che associa
# ogni sensore all'intero che lo rappresenta
def load_dataset(name, use_day_period=False, mapping=False):
    df = pd.read_csv(
        f"../dataset_csv/Ordonez{name}.csv", converters={"sensors": str}
    )

    # Discretizza le osservazioni dei sensori
    if use_day_period:
        df["sensors"] = df["sensors"] + df["period"].apply(str)

    df[["sensors"]] = df[["sensors"]].apply(lambda x: x.astype("category"))
    m = dict(enumerate(df["sensors"].cat.categories))
    df[["sensors"]] = df[["sensors"]].apply(lambda x: x.cat.codes)

    if mapping:
        return df, m

    return df


# --- Plotting ---
def plot_confusion_matrix(
    t, p, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    cm = sklearn.metrics.confusion_matrix(t, p)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def show_values(pc, fmt="%.2f", **kw):
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
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(
    AUC,
    title,
    xlabel,
    ylabel,
    xticklabels,
    yticklabels,
    figure_width=40,
    figure_height=20,
    correct_orientation=False,
    cmap="RdBu",
):
    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(
        AUC, edgecolors="k", linestyle="dashed", linewidths=0.2, cmap=cmap
    )

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
    plt.xlim((0, AUC.shape[1]))

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


def plot_classification_report(
    truth, predict, title="Classification report ", cmap="RdBu"
):
    classification_report = sklearn.metrics.classification_report(
        truth, predict
    )
    lines = classification_report.split("\n")

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1 : len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    xlabel = "Metrics"
    ylabel = "Classes"
    xticklabels = ["Precision", "Recall", "F1-score"]
    yticklabels = [
        "{0} ({1})".format(class_names[idx], sup)
        for idx, sup in enumerate(support)
    ]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(
        np.array(plotMat),
        title,
        xlabel,
        ylabel,
        xticklabels,
        yticklabels,
        figure_width,
        figure_height,
        correct_orientation,
        cmap=cmap,
    )
