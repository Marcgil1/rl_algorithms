import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as K

from .networks import Policy
from .networks import QValue
from .memory import Memory

class DDPGAgent:
    def __init__(self, env):
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        self.memory = Memory(env, int(1e6), 64)

        self.policy_opt = ko.Adam(lr=1e-4)
        self.qvalue_opt = ko.Adam(lr=1e-3)
        
        self.polyak = 0.001
        self.gamma  = 0.99

        self.policy      = Policy(env)
        self.policy_targ = Policy(env)
        self.qvalue      = QValue(env)
        self.qvalue_targ = QValue(env)
        self._init_target_nets()



    def act(self, obs, test=False):
        act   = self.policy.get_action(obs)
        if test:
            noise = np.zeros_like(act)
        else:
            noise = np.random.normal(0.0, 0.1, self.act_space.shape)

        return np.clip(
            act + noise,
            self.act_space.low,
            self.act_space.high
        )

    def observe(self, transition):
        self.memory.add(transition)
        
        if self.memory.can_sample:
            batch = self.memory.get_batch()
            self._update_qvalue(batch)
            self._update_policy(batch)
            self._update_target_nets()

    def _init_target_nets(self):
        obs = self.obs_space.sample()
        act = self.act_space.sample()

        self.policy.get_action(obs)
        self.qvalue.get_value(obs, act)
        self.policy_targ.get_action(obs)
        self.qvalue_targ.get_value(obs, act)
        
        self.policy_targ.set_weights(self.policy.get_weights())
        self.qvalue_targ.set_weights(self.qvalue.get_weights())

    def _update_target_nets(self):
        self.policy_targ.set_weights([
            self.polyak*w + (1 - self.polyak)*target_w
            for w, target_w in zip(
                self.policy.get_weights(),
                self.policy_targ.get_weights()
            )
        ])
        self.qvalue_targ.set_weights([
            self.polyak*w + (1 - self.polyak)*target_w
            for w, target_w in zip(
                self.qvalue.get_weights(),
                self.qvalue_targ.get_weights()
            )
        ])

    @tf.function
    def _update_qvalue(self, batch):
        obs1, acts, rews, obs2 = batch

        targets = (
            rews + self.gamma*self.qvalue_targ(obs2, self.policy_targ(obs2))
        )

        self.qvalue_opt.minimize(
            lambda: kls.MSE(targets, self.qvalue(obs1, acts)),
            self.qvalue.variables
        )
        
    @tf.function
    def _update_policy(self, batch):
        obs1, acts, rews, obs2 = batch

        self.policy_opt.minimize(
            lambda: -K.mean(self.qvalue(obs1, self.policy(obs1))),
            self.policy.variables
        )