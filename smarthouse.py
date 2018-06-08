import pandas as pd
import numpy as np
from ast import literal_eval
from pomegranate import HiddenMarkovModel, DiscreteDistribution, State


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
    for f in ['A', 'B']:
        df = pd.read_csv(f'dataset_csv/Ordonez{f}.csv',
            converters={'sensors': str})

        # Discretizza le osservazioni dei sensori
        df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
        mapping = dict(enumerate(df['sensors'].cat.categories))
        df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

        # TODO: Suddividere in train e test set
        size = int(df.shape[0]*0.8)
        trainset_s = df['activity'][:size]; testset_s = df['activity'][size:]
        trainset_o = df['sensors'][:size]; testset_o = df['sensors'][size:]

        P = prior(trainset_s)
        T = transition_matrix(trainset_s)
        O = obs_matrix(trainset_s, trainset_o)

        model = HiddenMarkovModel(name=f"model{f}")
        # Inizializzazione gli stati
        states = []
        for i in range(O.shape[0]):
            d = dict(enumerate(O[i,:]))
            states.append(State(DiscreteDistribution(d), name=f'{i}'))
        model.add_states(*states)

        # Definizione delle probabilità iniziali
        for i, p in enumerate(P):
            model.add_transition(model.start, states[i], p)

        # Definizione delle transizioni tra stati
        for i, s1 in enumerate(states):
            for j, s2 in enumerate(states):
                model.add_transition(s1, s2, T[i, j])

    #           (
    #            )
    #       __..---..__
    #   ,-='  /  |  \  `=-.
    #  :--..___________..--;
    #   \.,_____________,./
        model.bake()


    return P, T, O


if __name__ == '__main__':
    main()
