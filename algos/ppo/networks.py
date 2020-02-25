from tensorflow.keras.layers import (Layer,
                                     Dense,
                                     Conv2D,
                                     MaxPooling2D,
                                     Flatten)


class ConvolutionalBase(Layer):

    def __init__(self, activation='relu'):
        super(ConvolutionalBase, self).__init__()

        self.activation = activation

    def build(self, input_shape):
        super(ConvolutionalBase, self).build(input_shape)
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding='valid',
            activation=self.activation)
        self.pool1 = MaxPooling2D(
            pool_size=2)
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='valid',
            activation=self.activation)
        self.pool2 = MaxPooling2D(
            pool_size=2)
        self.flat1 = Flatten()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat1(x)
        return x


class DenseBase(Layer):

    def __init__(self, activation='relu'):
        super(DenseBase, self).__init__()

        self.activation = activation

    def build(self, input_shape):
        super(DenseBase, self).build(input_shape)
        self.dense1 = Dense(
            units=128,
            activation=self.activation)

    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        return x
