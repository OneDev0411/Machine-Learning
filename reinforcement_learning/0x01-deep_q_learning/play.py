#!/usr/bin/env python3
"""python script play.py that can display a game
played by the agent trained by train.py"""
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import gym
from rl.agents.dqn import DQNAgent
import keras as K
from keras.optimizers import Adam
train = __import__('train').train

env = gym.make('Breakout-v0')
train(env)
st = env.reset()
DQN = DQNAgent(
    model=K.models.load_model('policy.h5'),
    nb_actions=env.action_space.n,
    memory=SequentialMemory(limit=100000, window_length=3),
    policy=GreedyQPolicy())
DQN.compile(optimizer=Adam(
                  lr=1e-3,
                  clipnorm=1.0),)
DQN.test(
    env,
    visualize=True,
    nb_episodes=10,
)
