import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class Policy(tf.keras.Model):
    def __init__(self, env):
        super().__init__()
        self.act_dim  = len(env.action_space.shape)
        self.act_high = env.action_space.high
        self.act_low  = env.action_space.low

        self.dense1    = kl.Dense(
            units=64,
            activation='relu'
        )
        self.dense2    = kl.Dense(
            units=64,
            activation='relu'
        )
        self.dense3    = kl.Dense(
            units=self.act_dim,
            activation='sigmoid'
        )
        self.transform = kl.Lambda(
            lambda x: x * (self.act_high - self.act_low) + self.act_low
        )

    def call(self, obss):
        x = self.dense1(obss)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.transform(x)

        return x

    def get_action(self, obs):
        x = self.predict_on_batch(np.expand_dims(obs, axis=0))
        
        return x[0].numpy()


class QValue(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.concat = kl.Concatenate(
            axis=-1
        )
        self.dense1 = kl.Dense(
            units=64,
            activation='relu'
        )
        self.dense2 = kl.Dense(
            units=64,
            activation='relu'
        )
        self.dense3 = kl.Dense(
            units=1,
            activation='linear'
        )

    def call(self, obss, acts):
        x = self.concat([obss, acts])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

    def get_value(self, obs, act):
        x = self(
            np.expand_dims(obs, axis=0),
            np.expand_dims(act, axis=0)
        )

        return tf.squeeze(x).numpy()