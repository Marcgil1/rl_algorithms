from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from .networks import DenseBase

class Critic(Model):
    """
    Keras network representing a function of type observation_space -> (-inf, +inf).
    Inputs are expected to be batches of observations.
    """

    def __init__(self, observation_space):
        super(Critic, self).__init__()

        self.obs_space = observation_space

        self.base = DenseBase()
        self.last = Dense(
            units=1,
            activation='linear'
        )

    def call(self, inputs):
        x = self.base(inputs)
        x = self.last(x)
        return x