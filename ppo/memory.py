import numpy as np
from scipy.signal import lfilter

GAMMA = 0.99
GAE_LAMBDA = 0.95

class Memory:
    """
    Memory fitting PPO requeriments. i.e. stores tuples of the form
    (obs1, acts, rews, obs2, done, prob, vals). DOES NOT HAVE MAXIMUM
    SIZE => must be manually emptied.
    """
    
    def __init__(self):
        """
        Start memory.
        """
        self.buffer = []

    def store(self, transition):
        """
        Store transition.

        ARGS
        ----
        transition : iterable
            Must be of the form (obs1, acts, rews, obs2, done, prob, vals)
        """
        self.buffer.append(transition)

    def get_vals(self):
        """
        Return stored values + advantage estimates according to GAE.

        RETURNS
        -------
            tuple of (obs1, acts, rews, obs2, done, vals, prob, advs)
        """

        # Get stored values
        obs1 = np.array([row[0] for row in self.buffer])
        acts = np.array([row[1] for row in self.buffer])
        rews = np.array([row[2] for row in self.buffer])
        obs2 = np.array([row[3] for row in self.buffer])
        done = np.array([row[4] for row in self.buffer])
        prob = np.array([row[5] for row in self.buffer])
        vals = np.array([row[6] for row in self.buffer])

        # Not the most elegant piece of code ever, but it does the job.
        val1 = vals
        val2 = np.roll(vals, -1)
        val2[-1] = 0
        deltas = rews + GAMMA * val2 - val1
        advs = lfilter([1], [1, -GAMMA*GAE_LAMBDA], deltas[::-1])[::-1]

        # Normalise advantages.
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return obs1, acts, rews, obs2, done, vals, prob, advs

    def empty(self):
        """
        Forget everything.
        """
        self.buffer = []