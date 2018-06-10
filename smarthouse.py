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

        # TODO: Provare a fare la suddivisione a fine giornata
        # TODO: Provare a eseguire viterbi su sequenze più brevi...
        # ... magari anche su pomegrante
        size = int(df.shape[0]*0.75)
        trainset_s = df['activity'][:size]; testset_s = df['activity'].tolist()[size:]
        trainset_o = df['sensors'][:size]; testset_o = df['sensors'].tolist()[size:]

        P = prior(trainset_s)
        T = transition_matrix(trainset_s)
        O = obs_matrix(trainset_s, trainset_o)

        # Esegue l'algoritmo di Viterbi(1) sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        seq1, T1, T2 = viterbi(testset_o, T, O, P)
        c1 = 0
        for i, j in zip(seq1, testset_s):
            if i == j:
                c1 += 1
        print(f"Algoritmo 1, dataset {f}, trainset: {size}: {c1/len(seq1)}")

        # Esegue l'algoritmo di Viterbi(2) sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        seq2 = likeliest_path(P, T, O, testset_o)[0]
        c2 = 0
        for i, j in zip(seq2, testset_s):
            if i == j:
                c2 += 1
        print(f"Algoritmo 2, dataset {f}, trainset: {size}: {c2/len(seq2)}")

    return P, T, O


def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


def likeliest_path(initial, transition, emission, events):
    """Find the likeliest path in a hidden Markov Model resulting in the
    given events.

    Arguments:
    initial: arraylike(n) --- probability of starting in each state
    transition: arraylike(n, n) -- probability of transition between states
    emission: arraylike(n, e) -- probability of emitting each event in
        each state
    events -- iterable of events

    Returns:
    path: list(int) -- list of states in the most probable path
    p: float -- log-likelihood of that path

    """
    # Use log-likelihoods to avoid floating-point underflow. Note that
    # we want -inf for the log of zero, so suppress warnings here.
    with np.errstate(divide='ignore'):
        initial = np.log10(initial)
        transition = np.log10(transition)
        emission = np.log10(emission)

    # List of arrays giving most likely previous state for each state.
    prev = []

    events = iter(events)
    logprob = initial + emission[:, next(events)]
    for event in events:
        # p[i, j] is log-likelihood of being in state j, having come from i.
        p = logprob[:, np.newaxis] + transition + emission[:, event]
        prev.append(np.argmax(p, axis=0))
        logprob = np.max(p, axis=0)

    # Most likely final state.
    best_state = np.argmax(logprob)

    # Reconstruct path by following links and then reversing.
    state = best_state
    path = [state]
    for p in reversed(prev):
        state = p[state]
        path.append(state)
    return path[::-1], logprob[best_state]


if __name__ == '__main__':
    main()
