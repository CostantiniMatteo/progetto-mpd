import pandas as pd
import numpy as np
from ast import literal_eval


pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

def probability_distribution(seq1, seq2):
    n = 1 + max(seq1); m = 1 + max(seq2)
    M = np.zeros((n, m))

    # Conta delle occorrenze
    for s1, s2 in zip(seq1, seq2):
        M[s1][s2] += 1

    # Calcolo delle distribuzioni di probabilità
    row_sums = M.sum(axis=1)
    M = M / row_sums[:, np.newaxis]

    return M


# Calcola la distribuzione di probabilità degli stati
def prior(transitions):
    g = transitions.groupby(transitions)
    result = g.count()/g.count().sum()
    return result.as_matrix()


# Calcola la matrice di transizione data la sequenza di stati ad ogni tempo t
def transition_matrix(sequence):
    return probability_distribution(sequence, sequence[1:])


# Calcola la distribuzione di probabilità delle osservazioni per ogni stato
def obs_matrix(seq, obs):
    return probability_distribution(seq, obs)


def main():
    df = pd.read_csv('dataset_csv/OrdonezA.csv',
        converters={'sensors': str})

    df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
    mapping = dict(enumerate(df['sensors'].cat.categories))
    df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

    P = prior(df['activity'])
    T = transition_matrix(df['activity'])
    O = obs_matrix(df['activity'], df['sensors'])

    return P, T, O


if __name__ == '__main__':
    main()
