# Overview
This repo contains implementations for several RL algorithms with a focus on
simplicity. It is my intention that they serve as examples for people learning
RL. All algorithms are implemented in an ```Actor``` class presenting an
```act(obs)``` and ```observe(transition)``` interface.

Currently implemented algorithms are deep-learning based, though implementations
for classical algorithms are on the making... So far, the repo contains:
1. PPO, clip version
2. VPG
3. DDPG
4. TD3

# Resources
These resources have been (and continue being) quite useful:

- Spinning Up (spinningup.openai.com)
- Reinforcement Learning: An introduction
