import numpy as np
import pdb
import os
import torch
from torch import nn
from policies import NeuralNetwork, StablebaselinePolicy, D4RLPolicy
import warnings
import copy
import random
import gym

warnings.filterwarnings("error")

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_dataset_info(pathname):
    dataset_info = np.load('{}.npy'.format(pathname), allow_pickle = True).item()

    data = dataset_info['dataset']
    num_terminal_states = data['num_samples'] - np.count_nonzero(data['terminal_masks'])

    info = 'name: {}\n'.format(dataset_info['dataset_name'])\
        + 'batch size: {}\n'.format(dataset_info['batch_size'])\
        + 'traj length: {}\n'.format(dataset_info['traj_len'])\
        + 'oracle ret est: {}\n'.format(dataset_info['oracle_est_ret'])\
        + 'data ret est: {}\n'.format(dataset_info['data_est_ret'])\
        + 'pib ret est: {}\n'.format(dataset_info['pib_est_ret'])\
        + 'gamma: {}\n'.format(dataset_info['gamma'])\
        + 'samples: {}\n'.format(data['num_samples'])\
        + 'num endings: {}'.format(num_terminal_states)
    print (info)

def load_env(env_name):
    from gym.envs.registration import register
    if env_name == 'HumanoidStandup':
        register(
            id="CustomHumanoidStandup-v4",
            entry_point="custom_humstd:HumanoidStandup",
            max_episode_steps=1000,
        )
        env = gym.make('CustomHumanoidStandup-v4')
    elif env_name == 'Swimmer':
        register(
            id="CustomSwimmer-v4",
            entry_point="custom_swimmer:Swimmer",
            max_episode_steps=1000,
            reward_threshold=360.0,
        )
        env = gym.make('CustomSwimmer-v4')
    elif env_name == 'Cheetah':
        register(
            id="CustomCheetah-v4",
            entry_point="custom_cheetah:Cheetah",
            max_episode_steps=1000,
            reward_threshold=4800.0,
        )
        env = gym.make('CustomCheetah-v4')
    elif env_name == 'Hopper':
        env = gym.make('Hopper-v4')
    elif env_name == 'Walker':
        env = gym.make('Walker2d-v4')
    return env

def load_d4rl_policy(config, env, env_name, pi_num):
    dir_path = env_name.lower() + '/policies'
    pi_url = config[env_name]['d4rl']['pi_url'].format(pi_num)
    pi_filepath = dir_path + '/' + pi_url
    pi = D4RLPolicy(pi_filepath)
    return pi

def load_policies(config, env, env_name, pi_type = 'pib'):
    pi_num = config[env_name]['custom'][pi_type]
    algo = 'SAC'
    if env_name == 'HumanoidStandup':
        pi_path = 'humanoidstandup/policies/humstd_model_sac_{}_steps'.format(pi_num)
    elif env_name == 'Swimmer':
        pi_path = 'swimmer/policies/swimmer_model_sac_{}_steps'.format(pi_num)
    elif env_name == 'Cheetah':
        pi_path = 'cheetah/policies/cheetah_model_sac_{}_steps'.format(pi_num)

    pi = StablebaselinePolicy(name = 'MlpPolicy', algo = algo,\
        env = env, pretrained_path = pi_path, deterministic = False)
    return pi

def soft_target_update(online_net, target_net, tau = 0.005):
    online_params = online_net.state_dict()
    target_params = target_net.state_dict().items()
    for name, target_param in target_params:
        updated_params = tau * online_params[name] + (1. - tau) * target_param
        target_param.copy_(updated_params)

def collect_data_discrete(env, policy, num_trajectory, truncated_horizon, gamma = None):
    if not gamma:
        gamma = 1.
    phi = env.phi
    paths = []
    total_reward = 0.0
    densities = np.zeros((env.n_state, truncated_horizon))
    frequency = np.zeros(env.n_state)
    for i_trajectory in range(num_trajectory):
        path = {}
        path['obs'] = []
        path['acts'] = []
        path['rews'] = []
        path['nobs'] = []
        state = env.reset()
        sasr = []
        accum_gamma = np.ones(env.n_state)
        for i_t in range(truncated_horizon):
            action = policy(state)
            #p_action = policy[state, :]
            #action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward, done, _ = env.step(action)
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            #sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            densities[state, i_t] += 1
            total_reward += reward
            state = next_state
            if done:
                break
        paths.append(path)

    gammas = np.array([gamma ** i for i in range(truncated_horizon)])
    d_sum = np.sum(densities, axis = 0)
    densities = np.divide(densities, d_sum, out=np.zeros_like(densities), where = d_sum != 0)
    disc_densities = np.dot(densities, gammas)
    final_densities = (disc_densities / np.sum(gammas))
    return paths, frequency, total_reward / (num_trajectory * truncated_horizon), final_densities

def collect_data(env, policy, num_trajectory, truncated_horizon = None, use_true_latent = False, random_pi = False):
    paths = []
    num_samples = 0
    total_reward = 0.0
    all_s = []
    all_a = []
    for i_trajectory in range(num_trajectory):
        path = {}
        path['obs'] = []
        path['nobs'] = []
        path['acts'] = []
        path['rews'] = []
        path['dones'] = []
        state, _ = env.reset() # v4 gym outputs ob, {}
        sasr = []
        i_t = 0
        while True:
            if random_pi:
                action = env.action_space.sample()
            else:
                action = policy(env.convert_to_latents(state) if hasattr(env, 'convert_to_latents') and use_true_latent else state)
            next_state, reward, done, truncated, _ = env.step(action) # v4 changes
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            #sasr.append((state, action, next_state, reward))
            total_reward += reward
            state = next_state
            path['dones'].append(done)
            all_s.append(state)
            all_a.append(action)
            i_t += 1
            num_samples += 1
            #path['dones'].append(done or truncated)
            if done or truncated:
                break
            if truncated_horizon is not None and i_t >= truncated_horizon:
                break
        paths.append(path)

    # print ('online interaction')
    # print ('state')
    # print (np.mean(all_s,axis=0))
    # print (np.std(all_s, axis=0))
    # print ('action')
    # print (np.mean(all_a,axis=0))
    # print (np.std(all_a, axis=0))    
    return paths, total_reward / num_samples#(num_trajectory * truncated_horizon)

def collect_data_samples(env, policy, num_samples_to_collect, random_pi = False):
    paths = []
    num_samples = 0
    total_reward = 0.0
    all_s = []
    all_a = []

    collected_samples = 0
    num_trajs = 0
    while True:
        path = {}
        path['obs'] = []
        path['nobs'] = []
        path['acts'] = []
        path['rews'] = []
        path['dones'] = []
        state, _ = env.reset() # v4 gym outputs ob, {}
        while True:
            if random_pi:
                action = env.action_space.sample()
            else:
                action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action) # v4 changes
            path['obs'].append(state)
            path['acts'].append(action)
            path['rews'].append(reward)
            path['nobs'].append(next_state)
            total_reward += reward
            state = next_state
            path['dones'].append(done)
            all_s.append(state)
            all_a.append(action)
            collected_samples += 1
            if done or truncated:
                break
            
        num_trajs += 1
        if collected_samples >= num_samples_to_collect:
            break
        
        paths.append(path)
    
    print ('collected {} trajectories'.format(num_trajs))
    
    # print ('online interaction')
    # print ('state')
    # print (np.mean(all_s,axis=0))
    # print (np.std(all_s, axis=0))
    # print ('action')
    # print (np.mean(all_a,axis=0))
    # print (np.std(all_a, axis=0))
    return paths, total_reward / collected_samples#(num_trajectory * truncated_horizon)

def merge_datasets(data_list):
    merged_data = []
    for d in data_list:
        for traj in d:
            merged_data.append(traj)
    return merged_data 

def format_data_new(data, gamma, normalize_state = False):
    g_data = data['ground_data']

    s = []
    a = []
    sa = []
    sprime = []
    abs_s = []
    abs_sa = []
    abs_sprime = []
    rewards = []
    gammas = []
    terminal_masks = []
    policy_ratios = []
    for idx in range(len(g_data)):
        path = g_data[idx]
        obs = path['obs']
        nobs = path['nobs']
        acts = path['acts']
        rews = path['rews']
        dones = path['dones']
        accum_gamma = 1.
        for t in range(len(obs)):
            o = obs[t] / (255. if normalize_state else 1)
            no = nobs[t] / (255. if normalize_state else 1)
            a_pib = acts[t]
            r = rews[t]
            s.append(o)
            a.append(a_pib)
            sprime.append(no)
            rewards.append(r)
            gammas.append(accum_gamma)
            accum_gamma *= gamma
            terminal_masks.append(int((t < len(obs) - 1) and (not dones[t])))


    data = {
        'state_b': np.array(s),
        'action_b': np.array(a),
        'state_b_act_b': np.array(sa),
        'next_state_b': np.array(sprime),
        'rewards': np.array(rewards),
        'gammas': np.array(gammas),
        'init_state': data['initial_states'] / (255. if normalize_state else 1),
        'num_samples': len(s),
        'terminal_masks': np.array(terminal_masks),
        'true_g_policy_ratios': np.array(policy_ratios)
    }
    return data 

def get_err(true_val, pred_vals, metric = 'mse'):
    if metric == 'mse':
        error = np.square(np.array(true_val) - np.array(pred_vals))
    elif metric == 'abs':
        error = np.abs(np.array(true_val) - np.array(pred_vals))
    res = get_CI(error)
    return res

# statistics/visualization related
def get_CI(data, confidence = 0.95):

    if (np.array(data) == None).all():
        return {}
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    stats = {}
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    err = z * (std / np.sqrt(n))
    lower = mean - z * (std / np.sqrt(n))
    upper = mean + z * (std / np.sqrt(n))
    stats = {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'err': err,
        'max': np.max(data),
        'min': np.min(data)
    }
    return stats


