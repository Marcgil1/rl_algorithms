import numpy as np


class Memory:
    def __init__(self, env, max_size, batch_len):
        self.size      = 0
        self.max_size  = max_size
        self.batch_len = batch_len

        self.obs1 = np.zeros(
            (max_size,) + env.observation_space.shape,
            dtype=np.float32
        )
        self.acts = np.zeros(
            (max_size,) + env.action_space.shape,
            dtype=np.float32ç
        )
        self.rews = np.zeros(
            (max_size,),
            dtype=np.float32
        )
        self.obs2 = np.zeros(
            (max_size,) + env.observation_space.shape,
            dtype=np.float32
        )

        self.idx  = 0

    def add(self, transition):

        self.obs1[self.idx] = transition[0]
        self.acts[self.idx] = transition[1]
        self.rews[self.idx] = transition[2]
        self.obs2[self.idx] = transition[3]

        self.idx  = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self):
        idxs = np.random.randint(self.size, size=self.batch_len)

        obs1 = self.obs1[idxs]
        acts = self.acts[idxs]
        rews = self.rews[idxs]
        obs2 = self.obs2[idxs]

        return obs1, acts, rews, obs2

    @property
    def can_sample(self):
        return self.size >= self.batch_len

