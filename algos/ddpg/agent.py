import numpy as np
from .networks import Policy
from .networks import QValue
from .memory import Memory

class DDPGAgent:
    def __init__(self, env):
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        self.polyak = 0.01

        self.policy      = Policy(env)
        self.policy_targ = Policy(env)
        self.qvalue      = QValue()
        self.qvalue_targ = QValue()
        self._init_target_nets()

        self.memory = Memory(env, int(1e5),128)

    def act(self, obs, test=False):
        act   = self.policy.get_action(obs)
        if test:
            noise = np.zeros_like(act)
        else:
            noise = np.random.normal(0.0, 0.1, self.act_space.shape)

        return np.clip(
            act + noise,
            self.act_space.low,
            self.act_space.high
        )

    def observe(self, transition):
        self.memory.add(transition)
        # ...

    def _init_target_nets(self):
        obs = self.obs_space.sample()
        act = self.act_space.sample()
        self.policy.get_action(obs)
        self.qvalue.get_value(obs, act)
        self.policy_targ.get_action(obs)
        self.qvalue_targ.get_value(obs, act)
        
        self.policy_targ.set_weights(self.policy.get_weights())
        self.qvalue_targ.set_weights(self.qvalue.get_weights())

    def _update_target_nets(self):
        self.policy_targ.set_weights(
            self.polyak*self.policy.get_weights()
            + (1 - self.polyak)*self.policy_targ.get_weights()
        )
        self.qvalue_targ.set_weights(
            self.polyak*self.qvalue.get_weights()
            + (1 - self.polyak)*self.qvalue_targ.get_weights()
        )

if __name__ == '__main__':
    import gym
    
    env = gym.make('MountainCarContinuous-v2')
    
    agent = DDPGAgent(env)