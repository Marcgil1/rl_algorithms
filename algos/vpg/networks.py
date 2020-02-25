import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class ActionDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Value(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.dense = kl.Dense(
            units=32,
            activation='relu'
        )
        self.value = kl.Dense(
            units=1,
            activation='linear'
        )

    def call(self, obss):
        TFobss = tf.convert_to_tensor(obss)

        x = self.dense(TFobss)
        x = self.value(x)

        return x


class Policy(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.dense  = kl.Dense(
            units=32,
            activation='tanh'
        )
        self.logits = kl.Dense(
            units=env.action_space.n,
            activation='linear'
        )
        self.dist   = ActionDistribution()

    def call(self, obss):
        TFobss = tf.convert_to_tensor(obss)
        
        x = self.dense(TFobss)
        x = self.logits(x)
        
        return x

    def get_action(self, obs):
        logits = self.predict_on_batch(np.expand_dims(obs, axis=0))
        action = self.dist.predict_on_batch(logits)

        return tf.squeeze(action, axis=-1)
