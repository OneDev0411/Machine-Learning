#!/usr/bin/env python3
"""function that loads pre-made FrozenLakeEnv evnironment from OpenAI’s gym"""
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ desc: None or a list of lists containing a custom description
            of the map to load for the environment
        map_name: None or a string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery
        Returns: the environment """
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery)
