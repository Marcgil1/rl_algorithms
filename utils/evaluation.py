import numpy as np

def get_random_agent_reward(env, steps=50000):
    """
    Get the expected reward of following a random policy.

    ARGS
    ----
    env : gym.Env
        Environment considered.
    steps : int
        Number of steps for sampling.
    """
    step = 0
    total_rew = 0
    env.reset()
    while step < steps:
        # Interact.
        act = env.action_space.sample()
        _, rew, _, done = env.step(act)

        # Update counters.
        total_rew += rew
        step += 1
        if done:
            env.reset()

    return total_rew / steps

def get_batch(env, agent, batch_len):
    obss = np.empty((batch_len,) + env.observation_space.shape)
    acts = np.empty((batch_len,), dtype=np.int32)
    rets = np.empty((batch_len,))

    t   = 0
    obs = env.reset()
    for step in range(batch_len):
        obss[step] = obs.copy()
        acts[step] = agent.act(obs)
        
        obs, _, done, _ = env.step(acts[step])
        
        if done:
            rets[step - t: step] = t
            t   = 0
            obs = env.reset()
        else:
            t += 1

    rets[step - t: step] = t

    return obss, acts, rets

if __name__ == '__main__':
    import gym

    env = gym.make('MountainCarContinuous-v0')
    print(get_random_agent_reward(env))