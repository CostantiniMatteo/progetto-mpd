import numpy as np


def probability_distribution(seq1, seq2, n=None, m=None):
    if n is None:
        n = 1 + max(seq1)
    if m is None:
        m = 1 + max(seq2)
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
    if n is None:
        n = max(transitions) + 1
    g = transitions.groupby(transitions)
    result = g.count() / g.count().sum()
    return result.reindex(range(n)).fillna(0).as_matrix()


# Calcola la matrice di transizione data la sequenza di stati ad ogni tempo t
def transition_matrix(sequence, n=None, m=None):
    return probability_distribution(sequence, sequence[1:], n=n, m=m)


# Calcola la distribuzione di probabilità delle osservazioni per ogni stato
def obs_matrix(seq, obs, n=None, m=None):
    return probability_distribution(seq, obs, n=n, m=m)


# Genera una sequenza di stati e di osservazioni campionando utilizzando le
# distribuzioni di probabilità che definiscono la HMM.
def random_sample(P, T, O, n):
    assert n > 0
    states = []
    obs = []
    states.append(np.random.choice(range(len(P)), p=P))
    obs.append(np.random.choice(range(O.shape[1]), p=O[states[0]]))

    i = 0
    while i < n - 1:
        new_state = np.random.choice(range(len(P)), p=T[states[-1]])
        new_obs = np.random.choice(range(O.shape[1]), p=O[states[-1]])
        states.append(new_state)
        obs.append(new_obs)
        i += 1

    return states, obs


def hmm(state_seq, obs_seq, n=None, m=None):
    if n is None:
        n = max(state_seq) + 1
    if m is None:
        m = max(obs_seq) + 1

    P = prior(state_seq, n=n)
    T = transition_matrix(state_seq, n=n, m=n)
    O = obs_matrix(state_seq, obs_seq, n=n, m=m)

    return P, T, O


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
    with np.errstate(divide="ignore"):
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
