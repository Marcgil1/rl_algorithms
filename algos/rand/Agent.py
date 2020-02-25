class RandomAgent:

    def __init__(self, action_space, observation_space):
        self.action_space      = action_space
        self.observation_space = observation_space

    def act(self, obs):
        return self.action_space.sample()

    def observe(self, transition):
        pass