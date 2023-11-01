from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pdb
import argparse
import random
import os

import gym
import estimators
import utils
import yaml

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# saving
parser.add_argument('--outfile', default = None)

# common setup
parser.add_argument('--env_name', type = str, required = True)
parser.add_argument('--dataset_name', type = str)
parser.add_argument('--d4rl_dataset', type = str2bool)
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--oracle_num_trajs', default = 300, type = int)
parser.add_argument('--image_state', default = 'false', type = str2bool)
parser.add_argument('--samples_to_collect', default = 1e5, type = float)

# variables
parser.add_argument('--seed', default = 0, type = int)

FLAGS = parser.parse_args()

with open('cfg.yaml', 'r') as file:
    config = yaml.safe_load(file)

def env_setup():
    env = utils.load_env(FLAGS.env_name)
    return env

def policies_setup(env, pi_type = 'pib'):
    pi = utils.load_policies(config, env, FLAGS.env_name, pi_type = pi_type)
    return pi

def _data_prep(gamma, mdp, data_collection_info):

    data_list = []
    pi_names, pibs, sample_fracs = data_collection_info
    for pi_name, pi, sub_samples in zip(pi_names, pibs, sample_fracs):
        if sub_samples > 0:
            if pi_name == 'random':
                g_paths, _ = utils.collect_data_samples(mdp, None, sub_samples, random_pi = True)
            else:
                g_paths, _ = utils.collect_data_samples(mdp, pi, sub_samples)
            data_list.append(g_paths)

    merged_paths = utils.merge_datasets(data_list)
    initial_states = np.array([path['obs'][0] for path in merged_paths])
    data = {
        'initial_states': initial_states,
        'ground_data': merged_paths
    }
    # format data into relevant inputs needed by loss function
    data = utils.format_data_new(data, gamma)
    return data

def on_policy_estimate(num_trajs, gamma, mdp, pi, random_pi = False):
    # on-policy ground case
    g_pi_paths, _ = utils.collect_data(mdp, pi, num_trajs, random_pi = random_pi)
    g_pi_estimator = estimators.OnPolicy(pi, gamma)
    #g_pi_est_ret, g_pi_est_rew = g_pi_estimator.estimate(g_pi_paths)
    g_pi_est_ret = g_pi_estimator.estimate(g_pi_paths)
    return g_pi_est_ret#, g_pi_est_rew

def off_policy_estimate(data, gamma):
    estimator = estimators.OffPolicy(gamma)
    data_est_ret, data_est_rew = estimator.estimate(data)
    return data_est_ret, data_est_rew

def main():  # noqa
    seed = FLAGS.seed
    utils.set_seed_everywhere(seed)

    directory = 'datasets/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    mdp, gamma = env_setup(), FLAGS.gamma

    if FLAGS.d4rl_dataset:
        pie_num = config[FLAGS.env_name]['d4rl']['expert']
        pib_num = config[FLAGS.env_name]['d4rl']['medium']
        pie = utils.load_d4rl_policy(config, mdp, FLAGS.env_name, pie_num)
        pib = utils.load_d4rl_policy(config, mdp, FLAGS.env_name, pib_num)
        print ('loaded d4rl policies: {}, {}'.format(pie_num, pib_num))
    else:
        pie = policies_setup(mdp, pi_type = 'expert')
        pib = policies_setup(mdp, pi_type = 'medium')

    oracle_est_ret = on_policy_estimate(FLAGS.oracle_num_trajs, gamma, mdp, pie)
    print ('pie true (ret, rew): {} {}'.format(oracle_est_ret, oracle_est_ret * (1. - gamma)))
    
    pib_est_ret = on_policy_estimate(FLAGS.oracle_num_trajs, gamma, mdp, pib)
    print ('pib true (ret, rew): {} {}'.format(pib_est_ret, pib_est_ret * (1. - gamma)))

    random_est_ret = on_policy_estimate(FLAGS.oracle_num_trajs, gamma, mdp, None, random_pi = True)
    print ('rand true (ret, rew): {} {}'.format(random_est_ret, random_est_ret * (1. - gamma)))

    samples_to_collect = int(FLAGS.samples_to_collect)
    pib_names = ['expert', 'medium', 'random']
    pibs = [pie, pib, None]
    mix_ratios = config[FLAGS.env_name][FLAGS.dataset_name]
    sample_fracs = [int(ratio * samples_to_collect) for ratio in mix_ratios]
    print ('fraction: ' + str(sample_fracs))

    data_collection_info = (pib_names, pibs, sample_fracs)

    dataset_name_part = '{}-{}'.format('d4rl' if FLAGS.d4rl_dataset else 'custom', FLAGS.dataset_name)
    dataset_name = '{}K_name_{}'.format(int(samples_to_collect / 1000), dataset_name_part)

    ope_data = _data_prep(gamma, mdp, data_collection_info = data_collection_info)

    summary = {
        'dataset_name': '{}_{}'.format(FLAGS.env_name, dataset_name),
        'dataset': ope_data,
        'num_samples': samples_to_collect,
        'seed': seed,
        'pie_est_rew': oracle_est_ret * (1. - gamma),
        'pib_est_rew': pib_est_ret * (1. - gamma),
        'rand_est_rew': random_est_ret * (1. - gamma)
        #'data_est_rew': data_est_rew,
    }

    data_est_ret, data_est_rew = off_policy_estimate(ope_data, gamma)
    print ('collected dataset; data value (ret, rew): {} {}'.format(data_est_ret, data_est_rew))
    outfile = summary['dataset_name']
    np.save(directory + outfile, summary)
    print ('saved dataset')

if __name__ == '__main__':
    main()
