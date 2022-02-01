#!/usr/bin/env python3
"""function  that has the trained agent play an episode"""
import numpy as np
import gym


def play(env, Q, max_steps=100):
    """ env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode"""
    total_rewards = 0
    state = env.reset()
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state, :])
        new_state, reward, d, info = env.step(action)
        total_rewards += reward
        state = new_state
        if d:
            env.render()
            break
    env.close()
    return total_rewards
