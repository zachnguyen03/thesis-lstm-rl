import gym
from gym.spaces import Discrete, Box
import numpy as np
import random
import argparse

import gym, ray
from ray.rllib.agents import ppo
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

from ray.tune.registry import register_env

tf = try_import_tf()

class TextGenerationEnv():
    def __init__(self, config):
        self.observation_space = dataX[np.random.randint(0, len(dataX)-1)]
        self.action_space = config["chars"]
        self.token = None
        self.num_steps = 0
    
    def reset(self):
        self.token = np.random.randint(0, len(dataX)-1)
        self.num_steps = 0
        return self.token
    
    def step(self,action):
        if action == self.token:
            reward = 1
        else:
            reward = 0
        self.num_steps += 1
        done = self.num_steps > 100
        return 0, reward, done, {}
    

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1 if done else 0, done, {}

    
ray.shutdown()
ray.init()
trainer = ppo.PPOTrainer(env=SimpleCorridor, config={
    "env_config": {
        "chars": chars,      
    },  # config to pass to env class different lrs
    "num_workers": 1,  # parallelism
})

while True:
    print(trainer.train())
