import numpy as np


class Memory:
    def __init__(self, env, max_size, batch_len):
        self.size      = 0
        self.max_size  = max_size
        self.batch_len = batch_len

        self.obs1 = np.zeros((max_size,) + env.observation_space.shape)
        self.acts = np.zeros((max_size,) + env.action_space.shape)
        self.rews = np.zeros((max_size,))
        self.obs2 = np.zeros((max_size,) + env.observation_space.shape)

        self.idx  = 0

    def add(self, transition):

        self.obs1[self.idx] = transition[0]
        self.acts[self.idx] = transition[1]
        self.rews[self.idx] = transition[2]
        self.obs2[self.idx] = transition[3]

        self.idx  = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self):
        print(self.size)
        print(self.batch_len)
        idxs = np.random.randint(self.size, size=self.batch_len)

        obs1 = self.obs1[idxs]
        acts = self.acts[idxs]
        rews = self.rews[idxs]
        obs2 = self.obs2[idxs]

        return obs1, acts, rews, obs2

