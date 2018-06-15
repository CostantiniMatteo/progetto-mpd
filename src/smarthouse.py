import pandas as pd
import numpy as np
from functools import reduce
from utils import load_dataset
from hmm import hmm, viterbi, random_sample

pd.set_option("display.expand_frame_repr", False)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


# Divide il dataframe in training set e test set. Se tutto il dataset
# viene utilizzato per il training set, il test set restituito sarà vuoto
def trainset_testset(
    df,
    state="activity",
    obs="sensors",
    train_days=None,
    train_offset=0,
    test_days=None,
    test_offset=0,
):
    daylen = 24 * 60 * 60
    start = df.iloc[0]["timestamp"]
    end = df.iloc[-1]["timestamp"] + 60

    start_train = start + train_offset * daylen
    end_train = end if train_days is None else start_train + train_days * daylen
    start_test = end_train + test_offset * daylen
    end_test = (
        end if test_days is None else start_test + test_days * daylen + 60
    )

    trainset = df[
        (df["timestamp"] >= start_train) & (df["timestamp"] < end_train)
    ]
    testset = df[(df["timestamp"] >= start_test) & (df["timestamp"] < end_test)]

    return trainset[state], trainset[obs], testset[state], testset[obs]


# Separa il dataset in training set e test set e esegue l'algoritmo di viterbi
def smarthouse(
    dataset=["A", "B"],
    train_days=5,
    train_offset=0,
    test_days=None,
    test_offset=0,
    use_day_period=False,
    n_samples=None,
):
    if not (type(dataset) == tuple or type(dataset) == list):
        dataset = [dataset]

    truths = []
    predicts = []
    accs = []
    for f in dataset:
        df = load_dataset(f, use_day_period=use_day_period)

        train_s, train_o, test_s, test_o = trainset_testset(
            df,
            train_days=train_days,
            train_offset=train_offset,
            test_days=test_days,
            test_offset=test_offset,
        )

        # Calcolo delle distribuzioni della HMM
        n = max(df['activity'] + 1)
        m = max(df['sensors'] + 1)
        P, T, O = hmm(train_s, train_o, n=n, m=m)

        if n_samples:
            test_s, test_o = random_sample(P, T, O, n_samples)

        # Esegue l'algoritmo di Viterbi sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        predicted, p = viterbi(P, T, O, test_o)
        accuracy = reduce(
            lambda i, j: i + (1 if j[0] == j[1] else 0),
            zip(test_s, predicted),
            0,
        ) / len(predicted)

        accs.append(accuracy)
        truths.append(test_s)
        predicts.append(predicted)

    if len(accs) == 1:
        return truths[0], predicts[0], accs[0]

    return truths, predicts, accs


if __name__ == "__main__":
    t, p, a = smarthouse()
