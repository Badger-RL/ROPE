import torch
import numpy as np
import pdb
import random
import urllib.request
import os
import gym

class Dataset:
    def __init__(self, data,\
        normalize_states = False,\
        normalize_rewards = False,\
        normalize_actions = False,
        eps = 1e-5):

        dataset = data['dataset']

        self.curr_states = dataset['state_b'].astype(np.float32)
        self.curr_actions = dataset['action_b'].astype(np.float32)
        self.next_states = dataset['next_state_b'].astype(np.float32)
        self.rewards = dataset['rewards'].astype(np.float32)
        self.initial_states = dataset['init_state'].astype(np.float32)
        self.terminal_masks = dataset['terminal_masks'].astype(np.float32)
        self.num_samples = dataset['num_samples']
        self.batch_size = np.count_nonzero(1. - self.terminal_masks)
        self.eps = eps
        self.normalize_states_flag = normalize_states
        self.normalize_rewards_flag = normalize_rewards

        # TODO normalize actions??
        if normalize_states:
            self.state_mean = np.mean(self.curr_states, axis = 0)
            self.state_std = np.std(self.curr_states, axis = 0)

            self.curr_states = self.normalize_states(self.curr_states)
            self.next_states = self.normalize_states(self.next_states)
            self.initial_states = self.normalize_states(self.initial_states)
        else:
            self.state_mean = 0.
            self.state_std = 1.
        
        if normalize_rewards:
            self.reward_mean = np.mean(self.rewards)
            self.reward_std = np.std(self.rewards)

            self.rewards = self.normalize_rewards(self.rewards)
        else:
            self.reward_mean = 0.
            self.reward_std = 1.

        self.min_reward = np.min(self.rewards)
        self.max_reward = np.max(self.rewards)

        self.min_abs_reward_diff = 0
        self.max_abs_reward_diff = np.abs(self.max_reward - self.min_reward)

        #self.sarsa_data = self._get_sarsa_dataset()
    
    def _get_sarsa_dataset(self):

        curr_states = np.split(self.curr_states, self.batch_size)
        curr_actions = np.split(self.curr_actions, self.batch_size)
        next_states = np.split(self.next_states, self.batch_size)
        rewards = np.split(self.rewards, self.batch_size)

        c_s = []
        c_a = []
        n_s = []
        n_a = []
        r = []
        for s_, a_, r_, sn_ in zip(curr_states, curr_actions, rewards, next_states):
            an = a_[1:] # remove first
            new_sn = sn_[:-1] # remove last
            new_s = s_[:-1]
            new_a = a_[:-1]
            new_r = r_[:-1]

            c_s.append(new_s)
            c_a.append(new_a)
            n_s.append(new_sn)
            n_a.append(an)
            r.append(new_r)

        c_s = np.array(c_s)
        c_a = np.array(c_a)
        n_s = np.array(n_s)
        n_a = np.array(n_a)
        r = np.array(r)

        sarsa = {
            'curr_states': np.vstack(c_s),
            'curr_actions': np.vstack(c_a),
            'next_states': np.vstack(n_s),
            'next_actions': np.vstack(n_a),
            'rewards': np.hstack(r)
        }
        return sarsa

    def normalize_states(self, states):
        normalized_states = (states - self.state_mean) / np.maximum(self.eps, self.state_std)
        return normalized_states
    
    def unnormalize_states(self, normalized_states):
        states = normalized_states * np.maximum(self.eps, self.state_std) + self.state_mean
        return states

    def normalize_rewards(self, rewards):
        normalized_rewards = (rewards - self.reward_mean) / np.maximum(self.eps, self.reward_std)
        return normalized_rewards
    
    def unnormalize_rewards(self, normalized_rewards):
        rewards = normalized_rewards * np.maximum(self.eps, self.reward_std) + self.reward_mean
        return rewards
    
    def get_samples(self, mini_batch_size, sarsa = False):

        if sarsa:
            num_samples = self.sarsa_data['curr_states'].shape[0]
            mini_batch_size = min(mini_batch_size, num_samples)
            subsamples = np.random.choice(num_samples, mini_batch_size, replace = False)
            curr_states = self.sarsa_data['curr_states'][subsamples]
            curr_actions = self.sarsa_data['curr_actions'][subsamples]
            next_states = self.sarsa_data['next_states'][subsamples]
            next_actions = self.sarsa_data['next_actions'][subsamples]
            rewards = self.sarsa_data['rewards'][subsamples]
            initial_states = []
            terminal_masks = []
        else:
            mini_batch_size = min(mini_batch_size, self.num_samples)
            subsamples = np.random.choice(self.num_samples, mini_batch_size, replace = False)
            curr_states = self.curr_states[subsamples]
            curr_actions = self.curr_actions[subsamples]
            next_states = self.next_states[subsamples]
            rewards = self.rewards[subsamples]
            #initial_states = self.initial_states[subsamples]
            terminal_masks = self.terminal_masks[subsamples]
            next_actions = []

        sub_data = {
            'curr_states': curr_states,
            'curr_actions': curr_actions,
            'rewards': rewards,
            'next_states': next_states,
            'next_actions': next_actions,
            #'initial_states': initial_states,
            'terminal_masks': terminal_masks
        }
        return sub_data
    
    def get_initial_states_samples(self, mini_batch_size):
        # subsamples = np.random.choice(self.num_samples, mini_batch_size, replace = False)
        # initial_states = self.initial_states[subsamples]
        # return initial_states
        return self.initial_states
    
    def get_disjoint_batches(self, mini_batch_size):
        mini_batch_size = min(mini_batch_size, self.num_samples)
        indices = np.arange(self.num_samples)
        random.shuffle(indices)
        batched_indices = np.split(indices, self.num_samples / mini_batch_size)

        mini_batches = []
        for subsamples in batched_indices:
            curr_states = self.curr_states[subsamples]
            curr_actions = self.curr_actions[subsamples]
            next_states = self.next_states[subsamples]
            rewards = self.rewards[subsamples]
            #initial_states = self.initial_states[subsamples]
            terminal_masks = self.terminal_masks[subsamples]

            sub_data = {
                'curr_states': curr_states,
                'curr_actions': curr_actions,
                'rewards': rewards,
                'next_states': next_states,
                #'initial_states': initial_states,
                'terminal_masks': terminal_masks
            }
            mini_batches.append(sub_data)
        return mini_batches
