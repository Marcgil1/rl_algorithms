from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from .networks import DenseBase

class Actor(Model):
    """
    Keras network representing function of type
    observation_space -> [0, 1] ^ action_space
    Inputs are expected to be batches of actions.
    """

    def __init__(self, action_space, observation_space):
        """
        Initialise network.

        ARGS
        ----
        action_space : gym.spaces.Discrete
            Environment's action space. For the moment, just discrete spaces supported.

        observation_space : gym.spaces.Box
            Environment's observation space.
        """
        super(Actor, self).__init__()

        # Network parameters.
        self.base = DenseBase()
        self.last = Dense(
            units=action_space.n,
            activation='softmax')

    def call(self, inputs):
        """
        Get batch of probabilities.

        ARGS
        ----
        inputs : np.array
            Batch of observations.
        
        RETURNS
        -------
        np.array. Batch of tuples of length action_space.n. Each tuple
        represents a probability distribution over action_space.
        """
        x = self.base(inputs)
        x = self.last(x)
        return x