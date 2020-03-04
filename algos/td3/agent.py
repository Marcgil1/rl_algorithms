import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as K

from .memory import Memory
from .networks import Policy
from .networks import QValue


class TD3Agent:
    def __init__(self, env):
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        self.memory = Memory(env, int(1e6), 100)

        self.polyak              = 5e-3
        self.gamma               = 0.99
        self.act_noise           = 0.1
        self.targ_act_noise      = 0.2
        self.targ_act_clip       = 0.5
        self.policy_delay        = 2
        self.initial_exploration = 10000

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
        self.step += 1
        if self.step < self.initial_exploration:
            return self.act_space.sample()
        
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

        if self.memory.can_sample:
            batch = self.memory.get_batch()

            self._update_qvalues(batch)
            if not (self.step % self.policy_delay):
                self._update_policy(batch)
                self._update_target_nets()
            
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

    def _update_target_nets(self):
        self.qvalue1_targ.set_weights([
            self.polyak*w + (1 - self.polyak)*targ_w
            for w, targ_w in zip(
                self.qvalue1.get_weights(),
                self.qvalue1_targ.get_weights()
            )
        ])
        self.qvalue2_targ.set_weights([
            self.polyak*w + (1 - self.polyak)*targ_w
            for w, targ_w in zip(
                self.qvalue2.get_weights(),
                self.qvalue2_targ.get_weights()
            )
        ])
        self.policy_targ.set_weights([
            self.polyak*w + (1 - self.polyak)*targ_w
            for w, targ_w in zip(
                self.policy.get_weights(),
                self.policy_targ.get_weights()
            )
        ])

    @tf.function
    def _update_qvalues(self, batch):
        obs1, acts, rews, obs2 = batch

        targ_acts = self.policy_targ(obs2)
        noise     = tf.random.normal(
            shape=targ_acts.shape,
            mean=0.0,
            stddev=self.targ_act_noise
        )
        noise     = tf.clip_by_value(
            noise,
            -self.targ_act_clip,
            self.targ_act_clip,
        )
        targ_acts += tf.cast(noise, dtype=tf.float64)


        targets = tf.stop_gradient(
            rews +self.gamma*tf.minimum(
                self.qvalue1_targ(obs2, targ_acts),
                self.qvalue2_targ(obs2, targ_acts)
            )
        )

        with tf.GradientTape() as tape1:
            loss1 = kls.MSE(targets, self.qvalue1(obs1, acts))
        with tf.GradientTape() as tape2:
            loss2 = kls.MSE(targets, self.qvalue2(obs1, acts))
        grads1 = tape1.gradient(loss1, self.qvalue1.variables)
        grads2 = tape2.gradient(loss2, self.qvalue2.variables)

        grads_and_vars1 = list(zip(grads1, self.qvalue1.variables))
        grads_and_vars2 = list(zip(grads2, self.qvalue2.variables))

        self.qvalue_opt.apply_gradients(grads_and_vars1)
        self.qvalue_opt.apply_gradients(grads_and_vars2)

    @tf.function
    def _update_policy(self, batch):
        obs1, acts, rews, obs2 = batch

        with tf.GradientTape() as tape:
            loss = -K.mean(self.qvalue1(obs1, self.policy(obs1)))
        grads          = tape.gradient(loss, self.policy.variables)
        grads_and_vars = list(zip(grads, self.policy.variables))

        self.policy_opt.apply_gradients(grads_and_vars)