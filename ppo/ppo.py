import gym
import utils
from agent import Agent

ENV_NAME = "LunarLander-v2"
NUM_STEPS = int(1e5)
MAX_EP_LEN = 5000
RENDER = False
RENDER_FREQ = 300


if __name__ == '__main__':

    utils.setup_tf()
    utils.setup_keras()
    env = gym.make(ENV_NAME)

    agent = Agent(env.action_space, env.observation_space)

    episodes, step = 0, 0
    while step < NUM_STEPS:

        obs1, done, score, ep_step = env.reset(), False, 0, 0
        while not done and ep_step < MAX_EP_LEN:

            if RENDER and not step % RENDER_FREQ:
                env.render()
            act = agent.act(obs1)

            obs2, rew, done, _ = env.step(act)

            agent.observe((obs1, act, rew, obs2, done))
            score += rew
            obs1 = obs2
            step += 1
            ep_step += 1
        episodes += 1

        print("Episode: {}\tStep: {}\tReward: {}".format(episodes, step, score))