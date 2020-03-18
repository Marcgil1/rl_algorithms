import numpy as np
from collections import deque
from scipy.signal import lfilter
import sys

GAMMA = 0.99
GAE_LAMBDA = 0.95

class Memory:
    """
    Memory fitting PPO requeriments. i.e. stores tuples of the form
    (obs1, acts, rews, obs2, done, prob, vals).
    """
    
    def __init__(self, maxsize=400):
        """
        Start memory.
        """
        self.buffer   = deque(maxlen=maxsize)
        self.maxsize  = maxsize
        self.last_len = None

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

        # Get stored values.
        obs1 = np.array([row[0] for row in self.buffer], dtype=np.float32)
        acts = np.array([row[1] for row in self.buffer], dtype=np.int32)
        rews = np.array([row[2] for row in self.buffer], dtype=np.float32)
        obs2 = np.array([row[3] for row in self.buffer], dtype=np.float32)
        done = np.array([row[4] for row in self.buffer], dtype=np.bool)
        prob = np.array([row[5] for row in self.buffer], dtype=np.float32)
        vals = np.array([row[6] for row in self.buffer], dtype=np.float32)

        # Calculate advantages.
        val1 = vals
        val2 = np.roll(vals, -1)
        val2[-1] = 0
        deltas = rews + GAMMA * val2 - val1
        advs = lfilter([1], [1, -GAMMA*GAE_LAMBDA], deltas[::-1])[::-1]
        # Normalise advantages.
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = advs.astype(np.float32)

        return obs1, acts, rews, obs2, done, vals, prob, advs

    def empty(self):
        """
        Forget everything.
        """
        self.last_len = len(self.buffer)
        self.buffer   = deque(maxlen=self.maxsize)

    def size(self):
        if self.buffer:
            tr_size = sum(sys.getsizeof(elem) for elem in self.buffer[0])
        else:
            tr_size = 0
        return tr_size * len(self.buffer)