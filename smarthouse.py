import pandas as pd
import numpy as np
from datetime import datetime


pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

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


def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


def print_numpy_matrix(m):
    import sys
    np.savetxt(sys.stdout, m, '%6.4f')


# Genera una sequenza di stati e di osservazioni campionando utilizzando le
# distribuzioni di probabilità che definiscono la HMM.
def random_sample(P, T, O, n):
    assert(n > 0)
    states = []; obs = []
    states.append(np.random.choice(range(len(P)), p=P))
    obs.append(np.random.choice(range(O.shape[1]), p=O[states[0]]))

    i = 0
    while i < n:
        new_state = np.random.choice(range(len(P)), p=T[states[-1]])
        new_obs = np.random.choice(range(O.shape[1]), p=O[states[-1]])
        states.append(new_state); obs.append(new_obs)
        i += 1

    return states, obs


# TODO: Nice to have: leggere le osservazioni da un csv
#       ed eseguire Viterbi sulla sequenza letta
def main(train_rate=0.75, to_date=None, n_samples=0, length=60):
    res = []; res2 = []
    for f in ['A', 'B']:
        df = pd.read_csv(f'dataset_csv/Ordonez{f}.csv',
            converters={'sensors': str})

        # Discretizza le osservazioni dei sensori
        df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
        mapping = dict(enumerate(df['sensors'].cat.categories))
        df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

        # Divisione in testset e trainset
        if to_date:
            slice_at = to_date[f]
            trainset = df[df['timestamp'] < slice_at]
            testset = df[df['timestamp'] >= slice_at]
            trainset_s = trainset['activity']
            trainset_o = trainset['sensors']
            testset_s = testset['activity'].tolist()
            testset_o = testset['sensors'].tolist()
            size = trainset.shape[0]
        elif n_samples > 0:
            trainset_s = df['activity']
            trainset_o = df['sensors']
            size = trainset_s.shape[0]
        else:
            size = int(df.shape[0] * train_rate)
            trainset_s = df['activity'][:size]
            trainset_o = df['sensors'][:size]
            testset_s = df['activity'].tolist()[size:]
            testset_o = df['sensors'].tolist()[size:]
            print(f"Trainset: {trainset_s.shape[0]/df.shape[0]}")

        # Calcolo delle distribuzioni della HMM
        P = prior(trainset_s)
        T = transition_matrix(trainset_s)
        O = obs_matrix(trainset_s, trainset_o)

        if n_samples > 0:
            testset_s, testset_o = random_sample(P, T, O, n_samples)

        # Esegue l'algoritmo di Viterbi(1) sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        seq1, T1, T2 = viterbi(testset_o, T, O, P)
        c1 = 0
        for i, j in zip(seq1, testset_s):
            if i == j:
                c1 += 1
        print(f"Algoritmo 1, dataset {f}, trainset: {size}: {c1/len(seq1)}")
        res.append(c1/len(seq1))

        # Esegue l'algoritmo di Viterbi(2) sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        seq2, p = likeliest_path(P, T, O, testset_o)
        c2 = 0
        for i, j in zip(seq2, testset_s):
            if i == j:
                c2 += 1
        print(f"Algoritmo 2, dataset {f}, trainset: {size}: {c2/len(seq2)}")
        res2.append(c2/len(seq2))

    return res, res2


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
    main(n_samples=4000)
    import matplotlib.pyplot as plt
    xs = []; ys1 = []; ys2 = []

    # Plot facendo variare il trainset in percentuale
    # for i in range(50, 100, 1):
    #     res1, res2 = main(i/100)
    #     ys1.append(res1)
    #     ys2.append(res2)
    #     xs.append(i)

    # Plot facendo variare il trainset in giorni
    # start_A = date_to_timestamp("2011-11-28 00:00:00")
    # start_B = date_to_timestamp("2012-11-11 00:00:00")
    # for i in range(7, 14):
    #     d = {'A': start_A + 86400*i , 'B': start_B + 86400*i}
    #     try:
    #         res1, res2 = main(to_date=d)
    #         ys1.append(res1)
    #         ys2.append(res2)
    #         xs.append(i)
    #         print("Done", i)
    #     except:
    #         pass

    # plt.figure(1)
    # plt.plot(xs, [y_i[0] for y_i in ys1])
    # plt.plot(xs, [y_i[1] for y_i in ys1])
    # plt.figure(2)
    # plt.plot(xs, [y_i[0] for y_i in ys2])
    # plt.plot(xs, [y_i[1] for y_i in ys2])
    # plt.show()
