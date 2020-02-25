import numpy as np


class Memory:
    def __init__(self, env, batch_len=5000, gamma=0.99):
        self.obs_shape = env.observation_space.shape
        self.batch_len = batch_len
        self.gamma     = gamma
        self.idx       = 0
        self.step      = 0

        self._reset_buffers()

    def add(self, transition, val):
        if self.idx >= self.batch_len:
            return
        
        obs, act, rew, _, done = transition

        self.obss[self.idx]  = obs
        self.acts[self.idx]  = act
        self.rews[self.idx]  = rew
        self.vals[self.idx]  = val
        self.dones[self.idx] = done
        
        if done:
            rtgs, advs = self._get_rtgs_and_advs()
            self.rtgs[self.idx - self.step: self.idx] = rtgs
            self.advs[self.idx - self.step: self.idx] = advs
            self.step = 0
        else:
            self.step += 1
    
        self.idx  += 1
    
    def get(self):
        obss = self.obss
        acts = self.acts
        rtgs = self.rtgs
        advs = self.advs
        self._empty()
        return obss, acts, rtgs, advs

    def _reset_buffers(self):
        self.obss  = np.zeros((self.batch_len,) + self.obs_shape)
        self.acts  = np.zeros((self.batch_len,), dtype=np.int32)
        self.rews  = np.zeros((self.batch_len,))
        self.rtgs  = np.zeros((self.batch_len,))
        self.advs  = np.zeros((self.batch_len,))
        self.vals  = np.zeros((self.batch_len,))
        self.dones = np.zeros((self.batch_len,), dtype=np.bool)

    def _empty(self):
        self._reset_buffers()
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

    def _get_rtgs_and_advs(self):
        rtgs = self._get_rtgs()
        vals = self.vals[self.idx - self.step: self.idx]

        return rtgs, rtgs - vals

    def __len__(self):
        return self.idx