import torch
from torch import nn
import numpy as np
from policies import NeuralNetwork, CustomID
import copy
import time
import estimators
import pdb
from utils import soft_target_update

torch.autograd.set_detect_anomaly(True)

class ContinuousPrePhiFQE:
    def __init__(self,
        state_dims,
        action_dims,
        gamma,
        pie,
        abs_state_dims = None,
        abs_state_action_dim = None,
        q_hidden_dim = 32,
        q_hidden_layers = 1,
        activation = 'relu',
        Q_lr = 1e-5,
        q_reg = 0,
        phi = None,
        image_state = False,
        sa_phi = False,
        layer_norm = False,
        clip_target = False):

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.abs_state_dims = abs_state_dims
        self.pie = pie
        self.gamma = gamma
        self.q_hidden_dim = q_hidden_dim
        self.q_hidden_layers = q_hidden_layers
        self.activation = activation
        self.Q_lr = Q_lr
        self.q_reg = q_reg
        self.sa_phi = sa_phi
        self.clip_target = clip_target

        self.phi_defined = phi is not None

        q_input_dim = state_dims + action_dims
        if self.phi_defined:
            phi = phi.train(False)
            q_input_dim = abs_state_action_dim if self.sa_phi else abs_state_dims + action_dims
        q_output_dim = 1

        self.phi = phi if self.phi_defined else CustomID()

        if phi is None and image_state:
            return
        else:
        # if phi is defined use MLP with inputs as abs_state_dims
        # if phi is not defined but non-image-based then use MLP
            self.Q = NeuralNetwork(input_dims = q_input_dim,
                                output_dims = q_output_dim,
                                hidden_dim = q_hidden_dim,
                                hidden_layers = q_hidden_layers,
                                activation = self.activation,
                                layer_norm = layer_norm)

        self.Q_optimizer = torch.optim.AdamW(self.Q.parameters(), lr = self.Q_lr)#, weight_decay = 1e-5)
        self.target_Q = copy.deepcopy(self.Q)

    def train(self, data, epochs = 2000, print_log = True):
        
        mini_batch_size = 512
        min_obj = float('inf')
        best_epoch = -1

        curr_grad_norm = 0

        self.best_Q = copy.deepcopy(self.target_Q)
        disc_ret_ests = {}
        tr_losses = {}

        for epoch in range(epochs):
            sub_data = data.get_samples(mini_batch_size)
            curr_states = torch.Tensor(sub_data['curr_states'])
            curr_actions = torch.Tensor(sub_data['curr_actions'])
            curr_sa = torch.concat((curr_states, curr_actions), axis = 1)
            rewards = torch.Tensor(sub_data['rewards'])
            next_states = torch.Tensor(sub_data['next_states'])
            terminal_masks = torch.Tensor(sub_data['terminal_masks'])

            def _l2_reg(model):
                with torch.enable_grad():
                    l2_loss = torch.zeros(1)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(param, 2)))
                return l2_loss

            def _obj(curr_states, curr_actions, curr_sa, next_s, rews, term_masks):

                if self.sa_phi:
                    curr_sa = self.phi(curr_sa)
                else:
                    curr_sa = np.concatenate((self.phi(curr_states), curr_actions), axis = 1)

                q_curr_outputs = self.Q.forward(curr_sa).reshape(-1) / (1. - self.gamma)

                next_sampled_acts = torch.Tensor(self.pie.batch_sample(data.unnormalize_states(next_s)))
                if self.sa_phi:
                    next_sa = self.phi(torch.concat((next_s, next_sampled_acts), axis = 1))
                else:
                    next_sa = torch.concat((self.phi(next_s), next_sampled_acts), axis = 1)

                q_next_outputs = self.target_Q(next_sa).reshape(-1) / (1. - self.gamma) # no Q gradients

                target = torch.Tensor(rews + self.gamma * term_masks * q_next_outputs)

                if self.clip_target:
                    target = torch.clip(target,\
                        min = data.min_reward / (1. - self.gamma),\
                        max = data.max_reward / (1. - self.gamma))
                obj = torch.nn.functional.huber_loss(q_curr_outputs, target)
                reg = self.q_reg * _l2_reg(self.Q)
                total_obj = obj + reg
                objs = {
                    'obj': obj,
                    'reg': reg,
                    'total_obj': total_obj,
                }
                return objs 

            objs = _obj(curr_states, curr_actions, curr_sa, next_states, rewards, terminal_masks)
            total_obj = objs['total_obj']

            # clear gradients
            self.Q_optimizer.zero_grad()
            
            # compute gradients
            total_obj.backward()

            # processing gradients
            nn.utils.clip_grad_value_(self.Q.parameters(), clip_value = 1.0)

            # gradient step
            self.Q_optimizer.step()

            # soft target update
            soft_target_update(self.Q, self.target_Q)

            total_obj = objs['total_obj'].item()
            reg = objs['reg'].item()
            obj = objs['obj'].item()

            # making latest updated Q similiar to Ruosong's T round updates
            if True:
                min_obj = total_obj
                best_epoch = epoch
                self.best_Q = self.target_Q
            
            if epoch % 1000 == 0 or epoch == epochs - 1:
                init_ground_states = torch.Tensor(data.get_initial_states_samples(mini_batch_size))
                sampled_actions = torch.Tensor(self.pie.batch_sample(data.unnormalize_states(init_ground_states)))

                if self.sa_phi:
                    q_init_inputs = self.phi(torch.concat((init_ground_states, sampled_actions), axis = 1))
                else:
                    q_init_inputs = torch.concat((self.phi(init_ground_states), sampled_actions), axis = 1)
                
                ret = data.unnormalize_rewards(np.mean(self.best_Q(q_init_inputs))) / (1. - self.gamma)
                disc_ret_ests[epoch + 1] = ret 
                tr_losses[epoch + 1] = total_obj
                print ('epoch {}, loss {}, disc ret: {}'.format(epoch, [objs[name].item() for name in objs.keys()], ret))

        print ('best epoch {}, obj {}'.format(best_epoch, min_obj))
        self.disc_ret_ests = disc_ret_ests
        self.tr_losses = tr_losses

    def get_metrics(self):
        metrics = {
            'r_ests': self.disc_ret_ests,
            'tr_losses': self. tr_losses
        }
        return metrics

    def get_Q(self):
        return self.best_Q


    def get_phi(self):
        return self.phi
