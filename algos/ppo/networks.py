import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class ConvolutionalBase(kl.Layer):
    def __init__(self, activation='relu'):
        super(ConvolutionalBase, self).__init__()

        self.activation = activation

        self.conv1 = kl.Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding='valid',
            activation=self.activation
        )
        self.pool1 = kl.MaxPooling2D(
            pool_size=2
        )
        self.conv2 = kl.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='valid',
            activation=self.activation
        )
        self.pool2 = kl.MaxPooling2D(
            pool_size=2
        )
        self.flat1 = kl.Flatten()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat1(x)
        return x


class DenseBase(kl.Layer):
    def __init__(self, activation='relu'):
        super(DenseBase, self).__init__()

        self.activation = activation

        self.dense1 = kl.Dense(
            units=32,
            activation=self.activation
        )

    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        return x

class Policy(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        # Network parameters.
        self.base = DenseBase()
        self.last = kl.Dense(
            units=env.action_space.n,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.base(inputs)
        x = self.last(x)
        return x

    def get_probs(self, obs):
        obss = np.expand_dims(obs, 0)
        return self(obss)[0].numpy()

class Value(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.base = DenseBase()
        self.last = kl.Dense(
            units=1,
            activation='linear'
        )

    def call(self, inputs):
        x = self.base(inputs)
        x = self.last(x)
        return x

    def get_value(self, obs):
        obss = np.expand_dims(obs, 0)
        return self(obss)[0][0].numpy()