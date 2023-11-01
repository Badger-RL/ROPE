import gym
import numpy as np
import math
import pdb
from gym.envs.mujoco.swimmer_v4 import SwimmerEnv
from gym import logger
from gym.spaces import Box
from gym.envs.mujoco import MujocoEnv
import os 

class Swimmer(SwimmerEnv):
    def __init__(self):
        logger.set_level(50)
        super(Swimmer, self).__init__()
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float64
        )

        config_file = os.getcwd() + '/mujoco_xmls' + '/custom_swimmer.xml'
        MujocoEnv.__init__(self, config_file, 4, observation_space=observation_space)
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]

    # def step(self, action):
    #     obs, reward, done, truncated, info = super(Swimmer, self).step(action)
    #     #if hasattr(self, 'np_random') and self.np_random.random() < 0.03:
    #     #    obs, _ = self.reset()
    #     pdb.set_trace()
    #     return obs, reward, done, truncated, info#False, False, None

    def get_initial_state_samples(self, num_states):
        init_states = []
        for i in range(num_states):
            state, _ = self.reset() 
            init_states.append(state)
        return np.array(init_states)

    # abstraction related
    def phi(self, state):
        return np.array(state)

