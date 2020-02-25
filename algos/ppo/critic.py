from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from .networks import DenseBase

class Critic(Model):
    """
    Keras network representing a function of type observation_space -> (-inf, +inf).
    Inputs are expected to be batches of observations.
    """

    def __init__(self, observation_space):
        """
        Initialise Model.

        ARGS
        ----
        observation_space : gym.env.Box
            Observation space. Must be continuous.
        """
        super(Critic, self).__init__()

        # We might need this.
        self.obs_space = observation_space

        # Actual network parameters.
        self.base = DenseBase()
        self.last = Dense(
            units=1,
            activation='linear'
        )

    def call(self, inputs):
        """
        Return batch of values for batch of inputs.

        ARGS
        ----
        inputs : np.array
            Batch of observations.

        RETURNS
        -------
        Batch of values.
        """
        x = self.base(inputs)
        x = self.last(x)
        return x