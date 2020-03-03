import numpy as np
import tensorflow.keras.optimizers as ko

from .memory import Memory
from .networks import Policy
from .networks import QValue


class TD3Agent:
    def __init__(self, env):
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        self.memory = Memory(env, int(1e6), 100)

        self.polyak         = 5e-3
        self.gamma          = 0.99
        self.act_noise      = 0.1
        self.targ_act_noise = 0.2
        self.targ_act_clip  = 0.5
        self.policy_delay   = 2

        self.qvalue_opt = ko.Adam(1e-3)
        self.policy_opt = ko.Adam(1e-3)

        self.qvalue1      = QValue(env)
        self.qvalue2      = QValue(env)
        self.policy       = Policy(env)
        self.qvalue1_targ = QValue(env)
        self.qvalue2_targ = QValue(env)
        self.policy_targ  = Policy(env)
        self._init_target_nets()

        self.step = 0

    def act(self, obs, test=False):
        act = self.policy.get_action(obs)
        if test:
            noise = np.zeros_like(act)
        else:
            noise = np.random.normal(0, self.act_noise, act.shape)

        return np.clip(
            act + noise,
            self.act_space.low,
            self.act_space.high
        )

    def observe(self, transition):
        self.memory.add(transition)
        self.step += 1

        if self.memory.can_sample:
            batch = self.memory.get_batch()

            self._update_qvalues(batch)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if not self.step % self.policy_delay:
                self._update_actor(batch)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self._update_target_nets()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
    def _init_target_nets(self):
        obs = self.obs_space.sample()
        act = self.act_space.sample()

        self.qvalue1.get_value(obs, act)
        self.qvalue2.get_value(obs, act)
        self.policy.get_action(obs)
        self.qvalue1_targ.get_value(obs, act)
        self.qvalue2_targ.get_value(obs, act)
        self.policy_targ.get_action(obs)

        self.qvalue1_targ.set_weights(self.qvalue1.get_weights())
        self.qvalue2_targ.set_weights(self.qvalue2.get_weights())
        self.policy_targ.set_weights(self.policy.get_weights())
