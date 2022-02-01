#!/usr/bin/env python3
"""function that performs Q-learning"""
import numpy as np
import gym


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """ env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes """
    total_rewards = []
    epsilon_init = epsilon
    for episode in range(episodes):
        state = env.reset()
        current_reward = 0
        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(len(Q[state]))
            else:
                action = np.argmax(Q[state, :])
            new_state, reward, d, info = env.step(action)
            Q[state, action] = Q[state, action] * \
                (1 - alpha) + (alpha * (reward + gamma * np.max(Q[new_state])))
            state = new_state
            if d and reward == 0:
                reward = -1
            current_reward += reward
            if d:
                break
        epsilon = min_epsilon + (epsilon_init - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        total_rewards.append(current_reward)
    return Q, total_rewards
