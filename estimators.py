"""Implements different OPE methods."""
from __future__ import print_function
from __future__ import division

import pdb
import numpy as np
from utils import get_CI
import torch

class Estimator(object):

    def __init__(self):
        pass

    def estimate(self, paths):
        pass

class OffPolicy(Estimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def estimate(self, data):
        avg_rew = np.mean(data['rewards'])
        # return, avg rew
        return avg_rew / (1. - self.gamma), avg_rew

class OnPolicy(Estimator):

    def __init__(self, pi, gamma):
        self.pi = pi
        self.gamma = gamma

    def estimate(self, paths):
        m = len(paths) # number of trajectories
        total = 0
        total_normalization = 0
        ret = 0
        total_reward = 0
        num_rews = 0
        for path in paths:
            obs = path['obs']
            acts = path['acts']
            rews = path['rews']
            accum_gamma = 1.
            for t in range(len(obs)):
                o = obs[t]
                a = acts[t]
                r = rews[t]
                total += accum_gamma * r
                #total_normalization += accum_gamma
                accum_gamma *= self.gamma
                total_reward += r
                num_rews += 1

        avg_disc_ret = total / len(paths)
        #return total / total_normalization, total / len(paths)
        # average reward, averaged discounted return
        #return avg_disc_ret, total_reward / num_rews#avg_disc_ret * (1. - self.gamma)
        avg_rew = total_reward / num_rews
        #return avg_rew / (1. - self.gamma), avg_rew
        return avg_rew / (1. - self.gamma)


class QEstimate(Estimator):
    def __init__(self, Q, phi, pie, gamma, state_action_phi = False):
        self.Q = Q
        self.pie = pie
        self.phi = phi
        self.gamma = gamma
        self.state_action_phi = state_action_phi

    def estimate(self, data):
        init_states = torch.Tensor(data.initial_states)
        sampled_actions = torch.Tensor(self.pie.batch_sample(data.unnormalize_states(init_states)))
        if self.state_action_phi:
            q_init_input = self.phi(torch.concat((init_states, sampled_actions), axis = 1))
        else:
            q_init_input = torch.concat((self.phi(init_states), sampled_actions), axis = 1)
        # TODO handle discrete action case
        ret = data.unnormalize_rewards(np.mean(self.Q(q_init_input))) / (1. - self.gamma)
        return ret     

class Dice(Estimator):

    def __init__(self, ratios, policy_ratios = None):
        self.ratios = ratios
        self.policy_ratios = None
        if policy_ratios is not None:
            self.policy_ratios = policy_ratios.reshape(-1)

    def estimate(self, transition_tuples, rewards, gammas, temp = None, actions = None):
        ratios = self.ratios(transition_tuples)
        if actions is not None:
            ratios = ratios[np.arange(len(ratios)), actions]
        ratios = ratios.reshape(-1)
        if self.policy_ratios is not None:
            ratios = ratios * self.policy_ratios
        print (get_CI(ratios))
        #out = np.sum(ratios * rewards) / np.sum(ratios)
        out = np.mean(ratios * rewards) 
        return out

class DiscreteDice(Estimator):

    def __init__(self, ratios):
        self.ratios = ratios

    def estimate(self, data, rewards, gammas):
        ratios = self.ratios[np.argmax(data, axis = 1)]
        #out = np.mean(ratios * rewards)
        out = np.sum(ratios * rewards) / np.sum(ratios)
        return out

