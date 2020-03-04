import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class Policy(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.act_dim  = len(env.action_space.shape)
        self.act_high = env.action_space.high
        self.act_low  = env.action_space.low

        self.hidden1 = kl.Dense(
            units=256,
            activation='relu'
        )
        self.hidden2 = kl.Dense(
            units=256,
            activation='relu'
        )
        self.last_layer = kl.Dense(
            units=self.act_dim,
            activation='tanh'
        )
        self.transform = kl.Lambda(
            lambda x: (x + 1.)*(self.act_high - self.act_low)/2. + self.act_low
        )

    def call(self, obss):
        x = self.hidden1(obss)
        x = self.hidden2(x)
        x = self.last_layer(x)
        x = self.transform(x)

        return x

    def get_action(self, obs):
        x = self(
            np.expand_dims(obs, axis=0)
        )

        return x[0].numpy()


class QValue(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.concat     = kl.Concatenate()
        self.hidden1    = kl.Dense(
            units=256,
            activation='relu'
        )
        self.hidden2    = kl.Dense(
            units=256,
            activation='relu'
        )
        self.last_layer = kl.Dense(
            units=1,
            activation='linear'
        )
        self.reshape    = kl.Reshape(tuple())

    def call(self, obss, acts):
        x = self.concat([obss, acts])
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.last_layer(x)
        x = self.reshape(x)

        return x

    def get_value(self, obs, act):
        x = self(
            np.expand_dims(obs, axis=0),
            np.expand_dims(act, axis=0)
        )

        return tf.squeeze(x).numpy()