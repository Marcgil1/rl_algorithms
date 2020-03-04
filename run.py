import time
import numpy as np
import gym

from algos.vpg.agent import VPGAgent
from algos.ddpg.agent import DDPGAgent
from algos.td3.agent import TD3Agent


def test(env_fn, agent, trials=10, render=False):
    env  = env_fn()
    rews = np.zeros((trials,))

    for trial in range(trials):
        
        obs, done = env.reset(), False
        while not done:
            if render and trial == 0:
                env.render()
                time.sleep(0.02)
            act = agent.act(obs, test=True)
            obs, rew, done, _ = env.step(act)

            rews[trial] += rew
    env.close()
    return np.mean(rews)


def run_experiment(
        agent,
        env_fn,
        steps=int(1e6),
        batch_len=5000,
        test_freq=5000,
        test_fn=test
    ):

    env = env_fn()

    obs1 = env.reset()
    for step in range(steps):
        act = agent.act(obs1)

        obs2, rew, done, _ = env.step(act)

        agent.observe((obs1.copy(), act, rew, obs2.copy(), done))

        if done:
            obs1 = env.reset()
        else:
            obs1 = obs2

        if not step % test_freq:
            results = test_fn(env_fn, agent)
            print('Step: %d \tMean episode rew: %f' % (step + 1, results))
    env.close()


if __name__ == '__main__':

    env_name = 'HalfCheetah-v2'

    env = gym.make(env_name)
    agent = TD3Agent(env)

    run_experiment(
        agent,
        lambda: gym.make(env_name),
        test_fn=test
    )