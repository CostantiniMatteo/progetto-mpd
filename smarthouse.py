import pandas as pd
import numpy as np


pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=2)

# Calcola la matrice di transizione data la sequenza di stati ad ogni tempo t
def transition_matrix(sequence):
    # Numero di stati
    n = 1 + max(sequence)

    # Conta transizioni nel dataset
    T = np.zeros((n, n))
    for (i,j) in zip(sequence,sequence[1:]):
        T[i][j] += 1

    # Calcolo percentuali
    row_sums = T.sum(axis=1)
    T = T / row_sums[:, np.newaxis]

    return T


# Calcola la distribuzione di probabilità degli stati
def prior(transitions):
    g = transitions.groupby(transitions)
    result = g.count()/g.count().sum()
    return result.as_matrix()


# Calcola la distribuzione di probabilità delle osservazioni per ogni stato
def obs_prob(states, obs):
    pass


# TODO: Dei sensori ci interessa solo location per identificarli
# TODO: Cambiare la condizione del join: non sulla media, ma se le attività
#       e le rilevazioni dei sensori si accavallano
# TODO: Ad ogni tempo t viene associata l'osservazione di un solo sensore oppure
#       un vettore binario che indica per ogni sensore se è attivo?
def main():
    df = pd.read_csv('dataset_csv/OrdonezA.csv')
    T = transition_matrix(df['activity'])
    P = prior(df['activity'])
    O = obs_prob(df['activity', df['location']])


if __name__ == '__main__':
    main()
