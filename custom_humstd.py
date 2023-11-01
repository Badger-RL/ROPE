import gym
import numpy as np
import math
import pdb
from gym.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from gym import logger

class HumanoidStandup(HumanoidStandupEnv):
    def __init__(self):
        logger.set_level(50)
        super(HumanoidStandup, self).__init__()
        self.state_dims = self.observation_space.shape[0]
        self.action_dims = self.action_space.shape[0]

    # def step(self, action):
    #     obs, reward, done, truncated, info = super(HumanoidStandup, self).step(action)
    #     #if hasattr(self, 'np_random') and self.np_random.random() < 0.03:
    #     #    obs, _ = self.reset()
    #     return obs, reward, done, truncated, None#False, False, None

    def get_initial_state_samples(self, num_states):
        init_states = []
        for i in range(num_states):
            state, _ = self.reset() 
            init_states.append(state)
        return np.array(init_states)

    # abstraction related
    def phi(self, state):
        return np.array(state)

