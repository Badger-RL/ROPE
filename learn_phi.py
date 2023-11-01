import torch
from torch import nn
import numpy as np
from policies import NeuralNetwork
import copy
import time
import estimators
import pdb
import utils
from utils import soft_target_update

SQRT_EPS = 5e-5

class OffPolicySA:
    def __init__(self,
        ground_state_dims,
        action_dims,
        abs_state_action_dims,
        hidden_dim = 32,
        hidden_layers = 1,
        activation = 'relu',
        final_activation = None,
        layer_norm = False,
        lr = 3e-4,
        reg_param = 0,
        beta = 0.1,
        gamma = None,
        image_state = False,
        mdp = None,
        pie = None):

        self.ground_state_dims = ground_state_dims
        self.action_dims = action_dims
        self.abs_state_action_dims = abs_state_action_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.final_activation = final_activation
        self.lr = lr
        self.reg_param = reg_param
        self.gamma = gamma
        self.beta = beta
        self.mdp = mdp
        self.pie = pie

        if image_state:
            return
        else:
            self.phi = NeuralNetwork(input_dims = ground_state_dims + action_dims,
                                    output_dims = abs_state_action_dims,
                                    hidden_dim = hidden_dim,
                                    hidden_layers = hidden_layers,
                                    activation = self.activation,
                                    final_activation = self.final_activation,
                                    layer_norm = layer_norm)

        self.target_phi = copy.deepcopy(self.phi)
        self.optimizer = torch.optim.AdamW(self.phi.parameters(), lr = self.lr)

    def train(self, data, epochs = 2000, print_log = True):

        mini_batch_size = 512
        min_obj = float('inf')
        best_epoch = -1

        tr_losses = {}
        phi_mean_dim = {}
        phi_std_dim = {}

        self.best_phi = copy.deepcopy(self.target_phi)

        for epoch in range(epochs):
            sub_data = data.get_samples(mini_batch_size)
            curr_states = sub_data['curr_states']
            curr_actions = sub_data['curr_actions']
            curr_sa = np.concatenate((curr_states, curr_actions), axis = 1)
            rewards = sub_data['rewards']
            next_states = sub_data['next_states']
            pie_next_actions = self.pie.batch_sample(data.unnormalize_states(next_states))
            next_sa = np.concatenate((next_states, pie_next_actions), axis = 1)

            sub_data = data.get_samples(mini_batch_size)
            other_curr_states = sub_data['curr_states']
            other_curr_actions = sub_data['curr_actions']
            other_curr_sa = np.concatenate((other_curr_states, other_curr_actions), axis = 1)
            other_rewards = sub_data['rewards']
            other_next_states = sub_data['next_states']
            other_pie_next_actions = self.pie.batch_sample(data.unnormalize_states(other_next_states))
            other_next_sa = np.concatenate((other_next_states, other_pie_next_actions), axis = 1)

            def _orth_reg(model):
                with torch.enable_grad():
                    orth_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            param_flat = param.view(param.shape[0], -1)
                            sym = torch.mm(param_flat, torch.t(param_flat))
                            sym -= torch.eye(param_flat.shape[0])
                            orth_loss = orth_loss + sym.abs().sum()
                    return orth_loss

            def _l2_reg(model):
                with torch.enable_grad():
                    l2_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(param, 2)))
                return l2_loss

            def _obj(rews, other_rews,\
                state_actions, other_state_actions,\
                next_state_actions, other_next_state_actions):

                reward_dist = torch.Tensor(np.abs(rews - other_rews))
                phi_x = self.phi.forward(state_actions)
                phi_x_norm = torch.linalg.norm(phi_x, axis = 1)
                phi_y = torch.tensor(self.target_phi(other_state_actions)) # no gradients
                phi_y_norm = torch.linalg.norm(phi_y, axis = 1)
                cs = torch.sum(phi_x * phi_y, axis = 1) / (phi_x_norm * phi_y_norm)
                angle = torch.atan2(torch.sqrt(1. + SQRT_EPS - torch.square(cs)), cs)
                norm_avg = 0.5 * (torch.square(phi_x_norm) + torch.square(phi_y_norm))
                cs_dist = angle
                curr_Uxy = norm_avg + self.beta * cs_dist

                next_phi_x = torch.tensor(self.target_phi(next_state_actions))
                next_phi_x_norm = torch.linalg.norm(next_phi_x, axis = 1)
                next_phi_y = torch.tensor(self.target_phi(other_next_state_actions))
                next_phi_y_norm = torch.linalg.norm(next_phi_y, axis = 1)
                cs = torch.sum(next_phi_x * next_phi_y, axis = 1) / (next_phi_x_norm * next_phi_y_norm)
                angle = torch.atan2(torch.sqrt(1. + SQRT_EPS - torch.square(cs)), cs)
                next_norm_avg = 0.5 * (torch.square(next_phi_x_norm) + torch.square(next_phi_y_norm))
                next_cs_dist = angle
                next_Uxy = next_norm_avg + self.beta * next_cs_dist
                target = reward_dist + self.gamma * next_Uxy
                # target = torch.clip(target,\
                #         min = data.min_abs_reward_diff / (1. - self.gamma),
                #         max = data.max_abs_reward_diff / (1. - self.gamma))
                obj = torch.nn.functional.huber_loss(curr_Uxy, target)

                reg = self.reg_param * _l2_reg(self.phi)
                total_obj = obj + reg 
                objs = {
                    'obj': obj,
                    'reg': reg,
                    'total_obj': total_obj,
                    'norm_avg': torch.mean(norm_avg),
                    'cs_dist': torch.mean(cs_dist)
                }
                return objs

            objs = _obj(rewards, other_rewards,\
                        curr_sa, other_curr_sa,\
                        next_sa, other_next_sa)
            total_obj = objs['total_obj']

            # clear gradients
            self.optimizer.zero_grad()
            
            # compute gradients
            try:
                total_obj.backward()
            except:
                print ('error on backprop, breaking')
                break

            # processing gradients
            nn.utils.clip_grad_value_(self.phi.parameters(), clip_value = 1.0)
            
            # gradient step
            self.optimizer.step()

            # soft update target network
            soft_target_update(self.phi, self.target_phi)

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            if epoch % 1000 == 0 or epoch == epochs - 1:
                # TODO may have to an OPE step in here
                norm_avg = objs['norm_avg'].item()
                cs_dist = objs['cs_dist'].item()

                temp = self.best_phi(curr_sa)
                mean_dim = np.mean(np.mean(temp, axis=0))
                std_dim = np.mean(np.std(temp, axis=0))
                phi_mean_dim[epoch + 1] = mean_dim
                phi_std_dim[epoch + 1] = std_dim
                tr_losses[epoch + 1] = total_obj
                print ('epoch: {}, loss: {}, rep feature mean: {}, rep feature std: {}, norm avg: {}, cs dist: {}'.format(epoch, 
                    total_obj, 
                    mean_dim, 
                    std_dim, 
                    norm_avg,
                    cs_dist))

            if True:
                min_obj = total_obj
                self.best_phi = self.target_phi
                best_epoch = epoch

        self.tr_losses = tr_losses
        self.phi_mean_dim = phi_mean_dim
        self.phi_std_dim = phi_std_dim
        print ('best epoch {}, obj {}'.format(best_epoch, min_obj))

    def get_metrics(self):
        metrics = {
            'tr_losses': self.tr_losses,
            'phi_mean_dim': self.phi_mean_dim,
            'phi_std_dim': self.phi_std_dim
        }
        return metrics

    def get_phi(self):
        return self.best_phi
