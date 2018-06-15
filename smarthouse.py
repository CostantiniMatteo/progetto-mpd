import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime


pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

def probability_distribution(seq1, seq2, n=None, m=None):
    if n is None: n = 1 + max(seq1)
    if m is None: m = 1 + max(seq2)
    M = np.zeros((n, m))

    # Conta delle occorrenze
    for s1, s2 in zip(seq1, seq2):
        M[s1][s2] += 1

    # Pone a 'epsilon' le probabilità che sono zero
    M[M == 0] = 10e-2

    # Calcolo delle distribuzioni di probabilità
    row_sums = M.sum(axis=1)
    M = M / row_sums[:, np.newaxis]

    return M


# Calcola la distribuzione di probabilità degli stati
def prior(transitions, n=None):
    if n is None: n = max(transitions) + 1
    g = transitions.groupby(transitions)
    result = g.count()/g.count().sum()
    return result.reindex(range(n)).fillna(0).as_matrix()


# Calcola la matrice di transizione data la sequenza di stati ad ogni tempo t
def transition_matrix(sequence, n=None, m=None):
    return probability_distribution(sequence, sequence[1:], n=n, m=m)


# Calcola la distribuzione di probabilità delle osservazioni per ogni stato
def obs_matrix(seq, obs, n=None, m=None):
    return probability_distribution(seq, obs, n=n, m=m)


def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


def print_numpy_matrix(m):
    import sys
    np.savetxt(sys.stdout, m, '%6.4f')


def viterbi(initial, transition, emission, events):
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


# Genera una sequenza di stati e di osservazioni campionando utilizzando le
# distribuzioni di probabilità che definiscono la HMM.
def random_sample(P, T, O, n):
    assert(n > 0)
    states = []; obs = []
    states.append(np.random.choice(range(len(P)), p=P))
    obs.append(np.random.choice(range(O.shape[1]), p=O[states[0]]))

    i = 0
    while i < n - 1:
        new_state = np.random.choice(range(len(P)), p=T[states[-1]])
        new_obs = np.random.choice(range(O.shape[1]), p=O[states[-1]])
        states.append(new_state); obs.append(new_obs)
        i += 1

    return states, obs


def hmm(state_seq, obs_seq, n=None, m=None):
    if n is None: n = max(state_seq) + 1
    if m is None: m = max(obs_seq) + 1

    P = prior(state_seq, n=n)
    T = transition_matrix(state_seq, n=n, m=n)
    O = obs_matrix(state_seq, obs_seq, n=n, m=m)

    return P, T, O


def trainset_testset(df, state='activity', obs='sensors',
        train_days=None, train_offset=0, test_days=None, test_offset=0):
    daylen = 24 * 60 * 60
    start = df.iloc[0]['timestamp']; end = df.iloc[-1]['timestamp'] + 60

    start_train = start + train_offset * daylen
    end_train = end if train_days is None else start_train + train_days * daylen
    start_test = end_train + test_offset * daylen
    end_test = end if test_days is None else start_test + test_days * daylen + 60

    trainset = df[(df['timestamp'] >= start_train) & (df['timestamp'] < end_train)]
    testset = df[(df['timestamp'] >= start_test) & (df['timestamp'] < end_test)]

    return trainset[state], trainset[obs], testset[state], testset[obs]


def load_dataset(name, use_day_period=False, mapping=False):
    df = pd.read_csv(
        f'dataset_csv/Ordonez{name}.csv',
        converters={'sensors': str}
    )

    # Discretizza le osservazioni dei sensori
    if use_day_period:
        df['sensors'] = df['sensors'] + df['period'].apply(str)

    df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
    m = dict(enumerate(df['sensors'].cat.categories))
    df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

    if mapping:
        return df, m

    return df


def smarthouse(datasets=['A', 'B'], train_days=5, train_offset=0,
        test_days=None, test_offset=0, use_day_period=False, n_samples=None):

    if not (type(datasets) == tuple or type(datasets) == list):
        datasets = [datasets]

    truths = []; predicts = []; accs = []
    for f in datasets:
        df = load_dataset(f, use_day_period=use_day_period)

        train_s, train_o, test_s, test_o = trainset_testset(
            df, train_days=train_days, train_offset=train_offset,
            test_days=test_days, test_offset=test_offset
        )

        # Calcolo delle distribuzioni della HMM
        P, T, O = hmm(train_s, train_o)

        if n_samples:
            test_s, test_o = random_sample(P, T, O, n_samples)

        # Esegue l'algoritmo di Viterbi sul testset e calcola
        # calcola la percentuale di stati predetti correttamente
        predicted, p = viterbi(P, T, O, test_o)
        accuracy = reduce(lambda i, j: i + (1 if j[0] == j[1] else 0),
            zip(test_s, predicted), 0) / len(predicted)

        accs.append(accuracy)
        truths.append(test_s)
        predicts.append(predicted)

    if len(accs) == 1:
        return truths[0], predicts[0], accs[0]

    return truths, predicts, accs


if __name__ == '__main__':
    t, p, a = smarthouse()
    for i in range(len(a)):
        print(f'{a[i]:.3f}')
