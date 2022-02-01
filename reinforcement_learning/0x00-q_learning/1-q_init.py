#!/usr/bin/env python3
"""function that initializes the Q-table"""
import numpy as np
import gym


def q_init(env):
    """ env: the FrozenLakeEnv instance """
    return np.zeros((env.observation_space.n, env.action_space.n))
