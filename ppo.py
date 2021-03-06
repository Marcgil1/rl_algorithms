import datetime
import tensorflow as tf
import gym
import numpy as np
from utils.setup import (setup_tf,
                         setup_keras)
from algos.ppo.agent import PPOAgent

ENV_NAME    = "CartPole-v0"
NUM_STEPS   = int(1e6)
MAX_EP_LEN  = 200
RENDER      = False
RENDER_FREQ = 10
TEST_STEPS  = 10


# Tensorboard stuff.
advantages   = tf.keras.metrics.Mean('advantages',  dtype=tf.float32)
total_reward = tf.keras.metrics.Sum('total_reward', dtype=tf.float32)
mean_reward  = tf.keras.metrics.Mean('mean_reward', dtype=tf.float32)
action_std   = tf.keras.metrics.Mean('action_std',  dtype=tf.float32)

# Logger.
current_time   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir        = 'logs/ppo/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir) 

def record_data(info):
    advantages(info['advs'])
    total_reward(info['rews'])
    mean_reward(info['rews'])
    action_std(np.std(info['acts']))
    with summary_writer.as_default():
        tf.summary.scalar('advantages',   advantages.result(),   step=step)
        tf.summary.scalar('total_reward', total_reward.result(), step=step)
        tf.summary.scalar('mean_reward',  mean_reward.result(),  step=step)
        tf.summary.scalar('action_std',   action_std.result(),   step=step)
        advantages.reset_states()
        total_reward.reset_states()
        mean_reward.reset_states()
        action_std.reset_states()
    

if __name__ == '__main__':
    setup_tf()
    setup_keras()
    env = gym.make(ENV_NAME)

    agent = PPOAgent(env.action_space, env.observation_space)


    episodes, step = 0, 0
    while step < NUM_STEPS:

        obs1, done, score, ep_step = env.reset(), False, 0, 0
        while not done and ep_step < MAX_EP_LEN:

            if RENDER and not step % RENDER_FREQ:
                env.render()
            act = agent.act(obs1)

            obs2, rew, done, _ = env.step(act)

            agent.observe(
                (obs1, act, rew, obs2, done or ep_step + 1  >= MAX_EP_LEN))
            score += rew
            obs1 = obs2
            step += 1
            ep_step += 1
        episodes += 1

        # Take note
        record_data(agent.get_info())
        
        template = "Episode: {}\tStep: {}\tReward: {}"
        print(template.format(episodes, step, score))