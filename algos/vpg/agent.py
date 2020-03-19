import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from .networks import Policy
from .networks import Value
from .memory import Memory


class VPGAgent:
    def __init__(
            self,
            env,
            batch_len=5000,
            gamma=0.99
        ):
        self.batch_len    = batch_len
        self.memory       = Memory(env, batch_len, gamma)
        self.env          = env
        self.policy       = Policy(env)
        self.value        = Value()
        self.policy.compile(
            optimizer=ko.Adam(lr=1e-2),
            loss=self._policy_loss
        )
        self.value.compile(
            optimizer=ko.Adam(lr=1e-2),
            loss='mse'
        )

    def act(self, obs, test=False):
        act      = self.policy.get_action(obs).numpy()
        self.val = self.value.get_value(obs).numpy()

        # TODO: Sometimes act does not belong to action_space. Find out why.
        if act in self.env.action_space:
            return act
        else:
            return self.env.action_space.sample()

    def observe(self, transition):
        self.memory.add(transition, self.val)

        if len(self.memory) >= self.batch_len:
            self.train_on_batch(*self.memory.get())

    def train_on_batch(self, obss, acts, rtgs, advs):
        self.policy.train_on_batch(
            obss,
            np.concatenate([acts[:, None], advs[:, None]], axis=-1)
        )
        self.value.train_on_batch(
            obss,
            rtgs
        )

    def _policy_loss(self, acts_and_weights, logits):
        acts, weights = tf.split(acts_and_weights, 2, axis=-1)

        cce = kls.SparseCategoricalCrossentropy(from_logits=True)

        return cce(
            acts,
            logits,
            sample_weight=weights
        )