from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pdb
import argparse
import random
import yaml

import gym
import estimators
import utils
from fqe import ContinuousPrePhiFQE
from learn_phi import OffPolicySA
from behavior_dataset import Dataset

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
parser.add_argument('--d4rl_dataset', type = str2bool, default = 'false')
parser.add_argument('--d4rl_pi_num', type = int, default = 10)
parser.add_argument('--dataset_name', type = str)
parser.add_argument('--normalize_states', default = 'false', type = str2bool)
parser.add_argument('--normalize_rewards', default = 'false', type = str2bool)
parser.add_argument('--ope_method', type = str, default = 'fqe')
parser.add_argument('--encoder_name', type = str, default = 'identity')
parser.add_argument('--gamma', default = 0.999, type = float)
parser.add_argument('--epochs', default = 2000, type = int)
parser.add_argument('--phi_epochs', default = 50000, type = int)
parser.add_argument('--image_state', default = 'false', type = str2bool)
parser.add_argument('--beta', default = 0.1, type = float)
parser.add_argument('--aux_task', default = False, type = str2bool)

# variables
parser.add_argument('--seed', default = 0, type = int)
parser.add_argument('--samples_to_collect', default = 1e5, type = float)
parser.add_argument('--phi_lr', default = 3e-4, type = float)
parser.add_argument('--Q_lr', default = 1e-4, type = float)
parser.add_argument('--phi_outdim', default = 10, type = int)
parser.add_argument('--fqe_layer_norm', default = 'false', type = str2bool)
parser.add_argument('--rep_layer_norm', default = 'false', type = str2bool)
parser.add_argument('--phi_num_hidden_layers', default = 2, type = int)
parser.add_argument('--phi_hidden_dim', default = 64, type = int)
parser.add_argument('--Q_num_hidden_layers', default = 2, type = int)
parser.add_argument('--Q_hidden_dim', default = 64, type = int)
parser.add_argument('--fqe_clip_target', default = 'false', type = str2bool)

# misc
parser.add_argument('--exp_name', default = 'gan', type = str)
parser.add_argument('--print_log', default = 'false', type = str2bool)

FLAGS = parser.parse_args()

with open('cfg.yaml', 'r') as file:
    config = yaml.safe_load(file)

def env_setup():
    env = utils.load_env(FLAGS.env_name)
    return env

def policies_setup(env, pi_type = 'pib'):
    pi = utils.load_policies(config, env, FLAGS.env_name, pi_type = pi_type)
    return pi

def train_encoder(data, mdp, gamma, encoder_name = None, pie = None):
    if encoder_name == 'off-policy-sa':
        enc = OffPolicySA(ground_state_dims = mdp.observation_space.shape[0], abs_state_action_dims = FLAGS.phi_outdim, action_dims = mdp.action_space.shape[0],\
                                hidden_dim = FLAGS.phi_hidden_dim, hidden_layers = FLAGS.phi_num_hidden_layers, activation = 'relu', final_activation = None, layer_norm = FLAGS.rep_layer_norm,\
                                lr = FLAGS.phi_lr, gamma = gamma, pie = pie, image_state = FLAGS.image_state, mdp = mdp, beta = FLAGS.beta)
    elif 'target-phi' in encoder_name:
        if 'sa' in encoder_name:
            enc = pie.pi.critic_target.qf0[0:-1] # skip last layer
        else:
            enc = pie.policy.value_net # TODO
        phi = enc

    metrics = {}
    if 'target-phi' not in encoder_name:
        # train encoder
        print ('training encoder')
        enc.train(data, epochs = FLAGS.phi_epochs)
        phi = enc.get_phi()
        metrics = enc.get_metrics()
    return phi, metrics

def _common_phi_setup(enc_name, gamma, mdp, pie, pib, ope_data = None):

    phi_data = ope_data
    phi, metrics = train_encoder(phi_data, mdp, gamma, encoder_name = enc_name, pie = pie)
    return phi, metrics

def run_experiment_ope(ope_method, ope_data, gamma, mdp, pie, pib, phi = None, enc_name = None):
    if 'fqe' in ope_method:
        sa_phi = ('sa' in enc_name) or ('identity' in enc_name)
        if enc_name == 'target-phi-sa':
            FLAGS.phi_outdim = phi[-2].out_features
        abs_fqe = ContinuousPrePhiFQE(state_dims = mdp.observation_space.shape[0], action_dims = mdp.action_space.shape[0],
                                gamma = gamma, pie = pie,\
                                phi = phi, abs_state_dims = FLAGS.phi_outdim,\
                                q_hidden_layers = FLAGS.Q_num_hidden_layers, q_hidden_dim = FLAGS.Q_hidden_dim,\
                                activation = 'relu', 
                                Q_lr = FLAGS.Q_lr, image_state = FLAGS.image_state,
                                abs_state_action_dim = FLAGS.phi_outdim,
                                sa_phi = sa_phi, layer_norm = FLAGS.fqe_layer_norm,
                                clip_target = FLAGS.fqe_clip_target)
        abs_fqe.train(ope_data, epochs = FLAGS.epochs, print_log = FLAGS.print_log)
        metrics = abs_fqe.get_metrics()
        Q = abs_fqe.get_Q()
        phi = abs_fqe.get_phi()
        qestimator = estimators.QEstimate(Q, phi, pie, gamma, state_action_phi = sa_phi)
        # discounted return
        est = qestimator.estimate(ope_data)
    print ('est value: {}'.format(est))
    return est, metrics

def on_policy_estimate(batch_size, truncated_horizon, gamma, mdp, pi = None, random_pi = False):
    # on-policy ground case
    g_pi_paths, _ = utils.collect_data(mdp, pi, batch_size, truncated_horizon, random_pi = random_pi)
    g_pi_estimator = estimators.OnPolicy(pi, gamma)
    g_pi_est_ret = g_pi_estimator.estimate(g_pi_paths)
    return g_pi_est_ret

def main():  # noqa
    seed = FLAGS.seed
    utils.set_seed_everywhere(seed)

    mdp, gamma = env_setup(), FLAGS.gamma
    print ('gamma for evaluation: {}'.format(gamma))

    ope_method = FLAGS.ope_method
    enc_name = FLAGS.encoder_name

    if enc_name == 'target-phi-sa':
        FLAGS.normalize_states = False

    directory = 'datasets'
    samples_to_collect = int(FLAGS.samples_to_collect)
    dataset_name_part = '{}-{}'.format('d4rl' if FLAGS.d4rl_dataset else 'custom', FLAGS.dataset_name)
    dataset_name = '{}_{}K_name_{}'.format(FLAGS.env_name, int(samples_to_collect / 1000), dataset_name_part)
    dataset_info = np.load('{}/{}.npy'.format(directory, dataset_name), allow_pickle = True).item()
    if FLAGS.d4rl_dataset:
        pie = utils.load_d4rl_policy(config, mdp, FLAGS.env_name, FLAGS.d4rl_pi_num)
        pib = None
    else:
        pie = policies_setup(mdp, pi_type = 'expert')
        pib = policies_setup(mdp, pi_type = 'medium')

    # off policy data for OPE
    ope_data = dataset_info
    ope_data = Dataset(ope_data, normalize_states = FLAGS.normalize_states,\
        normalize_rewards = FLAGS.normalize_rewards)

    oracle_est_rew = dataset_info['pie_est_rew']
    oracle_est_ret = oracle_est_rew / (1. - gamma)
    print ('pie true: {} {}'.format(oracle_est_ret , oracle_est_rew))
    rand_est_rew = dataset_info['rand_est_rew']
    rand_est_ret = rand_est_rew / (1. - gamma)
    print ('rand true: {} {}'.format(rand_est_ret , rand_est_rew))
    data_est_ret = np.mean(ope_data.rewards) / (1. - gamma)
    print ('dataset true (ret, rew): {} {}'.format(data_est_ret , data_est_ret * (1. - gamma)))

    # train the encoder
    phi = None
    phi_metrics = {}
    if not FLAGS.aux_task:
        if 'identity' in enc_name:
            phi = None
        else:
            phi, phi_metrics = _common_phi_setup(enc_name, gamma, mdp, pie, pib, ope_data = ope_data)

    r_est, ope_metrics = run_experiment_ope(ope_method, ope_data, gamma, mdp, pie, pib, phi = phi, enc_name = enc_name)

    algo_name = '{}_{}'.format(FLAGS.ope_method, FLAGS.encoder_name)
    summary = {
        'env': FLAGS.env_name,
        'results': {
            FLAGS.dataset_name: {} # no name for the dataset
        },
        'encoder_name': FLAGS.encoder_name,
        'seed': seed,
        'hp': {
            'Q_lr': FLAGS.Q_lr,
            'phi_lr': FLAGS.phi_lr,
            'phi_outdim': FLAGS.phi_outdim,
            'beta': FLAGS.beta
        },
        'oracle_est': oracle_est_ret,
        'rand_est': rand_est_ret,
        'gamma': FLAGS.gamma,
        'normalize_states': FLAGS.normalize_states,
        'normalize_rewards': FLAGS.normalize_rewards
    }

    # encoder stats
    phi_tr_losses = []
    phi_mean_dim = []
    phi_std_dim = []
    if (enc_name == 'off-policy-sa'):
        for t in sorted(phi_metrics['tr_losses']):
            phi_tr_losses.append(phi_metrics['tr_losses'][t])
            phi_mean_dim.append(phi_metrics['phi_mean_dim'][t])
            phi_std_dim.append(phi_metrics['phi_std_dim'][t])

    # OPE stats
    ope_errs_training = []
    ope_tr_losses = []
    final_err = (utils.get_err([oracle_est_ret], [r_est], metric = 'abs')['mean'])
    rand_err = utils.get_err([oracle_est_ret], [rand_est_ret], metric = 'abs')['mean']
    final_rel_err = final_err / rand_err

    if 'r_ests' in ope_metrics:
        for t in sorted(ope_metrics['r_ests']):
            r = ope_metrics['r_ests'][t]
            rel_err = (utils.get_err([oracle_est_ret], [r], metric = 'abs')['mean']) / rand_err
            ope_errs_training.append(rel_err)
            ope_tr_losses.append(ope_metrics['tr_losses'][t])

    summary['results'][FLAGS.dataset_name][algo_name] = {
        'err': final_rel_err, # single MSE just for single trial
        'rand_err': rand_err,
        'final_err': final_err,
        'r_est': r_est,
        'rand_est': rand_est_ret,
        'errs_tr': ope_errs_training,
        'phi_tr_losses': phi_tr_losses,
        'ope_tr_losses': ope_tr_losses,
        'phi_mean_dim': phi_mean_dim,
        'phi_std_dim': phi_std_dim
    }
    print (summary)
    np.save(FLAGS.outfile, summary) 

if __name__ == '__main__':
    main()
