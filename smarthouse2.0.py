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
    for f in [ 'A', 'B']:
        df = pd.read_csv(f'dataset_csv/Ordonez{f}.csv',
            converters={'sensors': str})

        # Discretizza le osservazioni dei sensori
        df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
        mapping = dict(enumerate(df['sensors'].cat.categories))
        df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

        # TODO: Suddividere in train e test set
        size = int(df.shape[0]*0.75)
        trainset_s = df['activity'][:size]; testset_s = df['activity'].tolist()[size:]
        trainset_o = df['sensors'][:size]; testset_o = df['sensors'].tolist()[size:]

        P = prior(trainset_s)
        T = transition_matrix(trainset_s)
        O = obs_matrix(trainset_s, trainset_o)


        seq = likeliest_path(P, T, O, testset_o)[0]
        
        c = 0
        for i, j in zip(seq, testset_s):
            if i == j:
                c += 1

        print(c/len(seq))

            

    return P, T, O


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
