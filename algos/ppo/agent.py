import tensorflow as tf
import numpy as np

from .actor import Actor
from .critic import Critic
from .memory import Memory

EPSILON = 0.2
ACTOR_UPDATE_STEPS = 10
CRITIC_UPDATE_STEPS = 10

class PPOAgent:
    """
    Implementation of PPO agent structured as an actor-critic method.
    """
    
    def __init__(self, env):
        """
        Initialise agent.

        ARGS
        ----
        action_space : gym.spaces.Discrete
            Environment's action space. By now, just discrete action spaces are
            supported.
        observation_space : gym.spaces.Box
            Environment's observation space. Must be continuous.
        """

        # Store environmental spaces.
        self.act_space = env.action_space
        self.obs_space = env.observation_space

        # Memory and networks.
        self.memory       = Memory()
        self.actor        = Actor(self.act_space, self.obs_space)
        self.critic       = Critic(self.obs_space)
        self.actor_loss   = None
        self.critic_loss  = None
        self.advs         = None
        self.rews         = None
        self.acts         = None
        self.probs        = None
        self.val          = None

        # Optimizers
        self.actor_optimizer  = tf.keras.optimizers.Adam(0.005)
        self.critic_optimizer = tf.keras.optimizers.Adam(0.005)

    def act(self, obs, test=False):
        """
        Take action acordint to obs.

        ARGS
        ----
        obs : np.array
            Observation

        RETURNS
        -------
            action belonging to env.action_space
        """

        self.probs = self.actor(np.expand_dims(obs, 0))[0].numpy()
        self.val   = self.critic(np.expand_dims(obs, 0))[0][0].numpy()

        return np.random.choice(self.act_space.n, p=self.probs)

    def observe(self, transition):
        """
        Observe a transition.

        ARGS
        ----
            transition : iterable
                Representation of a transition in the environment. Must
                be of the form (obs1, act, rew, obs2, done).
        """
        self.memory.store(list(transition) + [self.probs, self.val])

        if transition[-1]:
            self._update()

    def get_info(self):
        return {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
            'advs': self.advs,
            'rews': self.rews,
            'acts':self.acts
        }

    def _update(self):
        obs1, acts, rews, obs2, done, vals, probs, advs = self.memory.get_vals()
        
        self.actor_loss  = self._update_actor(obs1, acts, advs, probs)
        self.critic_loss = self._update_critic(obs1, rews)
        self.advs        = advs
        self.rews        = rews
        self.acts        = acts

        self.memory.empty()

    def _update_actor(self, obs1, acts, advs, probs):
        
        for _ in range(ACTOR_UPDATE_STEPS):
            with tf.GradientTape() as tape:
                losses = actor_loss(self.actor(obs1), obs1, acts, advs, probs)

            grads = tape.gradient(losses, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(grads, self.actor.trainable_variables))
        return np.mean(losses)

    def _update_critic(self, obs1, rews):
        for _ in range(CRITIC_UPDATE_STEPS):
            with tf.GradientTape() as tape:
                vals = self.critic(obs1)
                losses = tf.keras.losses.MSE(vals, rews)

            grads = tape.gradient(losses, self.critic.trainable_variables) # Add grads records
            self.critic_optimizer.apply_gradients(
                zip(grads,self.critic.trainable_variables))
        return np.mean(losses)

@tf.function
def actor_loss(logits, obs1, acts, advs, old_probs):

    indices = [
        [i, acts[i]]
        for i in range(len(obs1))
    ]

    # Get probabilities of taken actions.
    probs = tf.gather_nd(
        logits,
        tf.convert_to_tensor(indices))

    # Get probabilities of actions which actually got taken.
    old_probs = tf.gather_nd(
        old_probs,
        tf.convert_to_tensor(indices))
    ratios = probs / (old_probs + 1e-8)
    clip_probs = tf.clip_by_value(ratios, 1.-EPSILON, 1.+EPSILON)

    # PPO loss.
    loss = -tf.reduce_mean(
        tf.minimum(
            ratios*advs,
            clip_probs*advs))
    return loss