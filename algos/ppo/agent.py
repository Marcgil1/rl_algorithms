import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls

from .networks import Policy
from .networks import Value
from .memory import Memory

EPSILON = 0.2
ACTOR_UPDATE_STEPS = 10
CRITIC_UPDATE_STEPS = 10


class PPOAgent:
    def __init__(self, env):
        # Store environmental spaces.
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        # Memory and networks.
        self.memory = Memory()
        self.policy = Policy(env)
        self.value  = Value(env)

        self.policy_opt = ko.Adam(0.005)
        self.value_opt  = ko.Adam(0.005)

        self.probs = None
        self.val   = None

    def act(self, obs, test=False):
        self.probs = self.policy.get_probs(obs)
        self.val   = self.value.get_value(obs)

        return np.random.choice(self.act_space.n, p=self.probs)

    def observe(self, transition):
        self.memory.store(list(transition) + [self.probs, self.val])

        if transition[-1]:
            self._update()

    def _update(self):
        obs1, acts, rews, _, _, _, probs, advs = self.memory.get_vals()

        self._update_policy(obs1, acts, advs, probs)
        self._update_value(obs1, rews)
        
        self.memory.empty()

    def _update_policy(self, obs1, acts, advs, probs):
        for _ in range(ACTOR_UPDATE_STEPS):
            self.policy_opt.minimize(
                lambda: self._actor_loss(self.policy(obs1), obs1, acts, advs, probs),
                self.policy.trainable_weights
            )
    
    def _update_value(self, obs1, rews):
        for _ in range(CRITIC_UPDATE_STEPS):
            self.value_opt.minimize(
                lambda: kls.MSE(rews, self.value(obs1)),
                self.value.trainable_weights
            )
    
    @staticmethod
    def _actor_loss(probs, obs1, acts, advs, old_probs):
        indices = [
            [i, acts[i]]
            for i in range(len(obs1))
        ]

        # Get probabilities of taken actions.
        probs = tf.gather_nd(
            probs,
            tf.convert_to_tensor(indices)
        )

        # Get probabilities of actions which actually got taken.
        old_probs = tf.gather_nd(
            old_probs,
            tf.convert_to_tensor(indices)
        )
        ratios = probs / (old_probs + 1e-8)
        clip_probs = tf.clip_by_value(ratios, 1.-EPSILON, 1.+EPSILON)

        # PPO loss.
        loss = -tf.reduce_mean(
            tf.minimum(
                ratios*advs,
                clip_probs*advs
            )
        )
        return loss