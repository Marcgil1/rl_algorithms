import numpy as np


class Memory:
    def __init__(self, env, batch_len=5000, gamma=0.99):
        self.obs_shape = env.observation_space.shape
        self.batch_len = batch_len
        self.gamma     = gamma
        self.obss      = np.zeros((batch_len,) + self.obs_shape)
        self.acts      = np.zeros((batch_len,), dtype=np.int32)
        self.rews      = np.zeros((batch_len,))
        self.rtgs      = np.zeros((batch_len,))
        self.dones     = np.zeros((batch_len,), dtype=np.bool)
        self.idx       = 0
        self.step      = 0

    def add(self, transition,):
        if self.idx >= self.batch_len:
            return
        
        obs, act, rew, _, done = transition

        self.obss[self.idx]  = obs
        self.acts[self.idx]  = act
        self.rews[self.idx]  = rew
        self.dones[self.idx] = done
        
        if done:
            self.rtgs[self.idx - self.step: self.idx] = self._get_rtgs()
            self.step = 0
        else:
            self.step += 1
    
        self.idx  += 1
    
    def get(self):
        obss, acts, rtgs = self.obss, self.acts, self.rtgs
        self._empty()
        return obss, acts, rtgs

    def _empty(self):
        self.obss  = np.zeros((self.batch_len,) + self.obs_shape)
        self.acts  = np.zeros((self.batch_len,), dtype=np.int32)
        self.rews  = np.zeros((self.batch_len,))
        self.rtgs  = np.zeros((self.batch_len,))
        self.dones = np.zeros((self.batch_len,), dtype=np.bool)
        self.idx   = 0
        self.step  = 0

    def _get_rtgs(self):
        rews = self.rews[self.idx - self.step: self.idx]
        rtgs = np.append(
            np.zeros_like(rews),
            0
        )

        for t in reversed(range(rews.shape[0])):
            rtgs[t] = rews[t] + self.gamma * rtgs[t + 1] * (1 - self.dones[t])

        return rtgs[:-1]


    def __len__(self):
        return self.idx