import torch
from torch import nn
import numpy as np
import pickle
import pdb

from stable_baselines3 import SAC

class PositivityActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _positivity_activation(self, input):
        return torch.square(input)
        #return torch.log(1 + torch.exp(input))

    def forward(self, input):
        return self._positivity_activation(input)

class Symlog(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _symlog(self, input):
        return torch.sign(input) * torch.log(torch.abs(input) + 1)

    def forward(self, input):
        return self._symlog(input) 

class CustomID(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()

    def __call__(self, s):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        out = self.id(s)
        if torch.is_tensor(out):
            out = out.detach().numpy()
        return out

    def forward(self, s, requires_grad = True):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        out = self.id(s)
        return out

class CustomFlatten(nn.Module):
    def forward(self, inputs):
        sh = inputs.shape
        if len(sh) == 3: # single image with dims [channel, width, height]
            out = torch.flatten(inputs, start_dim = 0)
        elif len(sh) == 4: # batch of images [batch_size, channel, width, height]
            out = torch.flatten(inputs, start_dim = 1)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dim = 16, hidden_layers = 1,
                    activation = 'tanh',
                    batch_norm = False,
                    convnet_prefix = None,
                    final_activation = None,
                    layer_norm = False):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.tensor = torch.as_tensor
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = self._get_act_fn(activation)
        self.final_activation = self._get_act_fn(final_activation)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.convnet_prefix = convnet_prefix
        self.penultimate, self.output = self._create_network()
        self._initialize()

    def _get_act_fn(self, activation):
        th_act = None
        if activation == 'tanh':
            th_act = nn.Tanh()
        elif activation == 'relu':
            th_act = nn.ReLU()
        elif activation == 'silu':
            th_act = nn.SiLU()
        elif activation == 'lrelu':
            th_act = nn.LeakyReLU()
        elif activation == 'positivity':
            th_act = PositivityActivation()
        elif activation == 'sigmoid':
            th_act = nn.Sigmoid()
        elif activation == 'symlog':
            th_act = Symlog()
        return th_act

    def __call__(self, s):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        if self.convnet_prefix:
            s = self.convnet_prefix.forward(s)
        net_out = self.output(s)
        net_out = net_out.detach().numpy()
        return net_out

    def forward(self, s, requires_grad = True):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        if self.convnet_prefix:
            s = self.convnet_prefix.forward(s)
        net_out = self.output(s)
        return net_out

    def get_penultimate(self, s, requires_grad = True):
        s = torch.from_numpy(s).float()
        pen = self.penultimate(s)
        return pen

    def _create_network(self):
        net_arch = []
        curr_dims = self.input_dims
        next_dims = self.hidden_dim

        for l in range(self.hidden_layers):
            net_arch.append(nn.Linear(curr_dims, next_dims))
            if self.batch_norm:
                net_arch.append(nn.BatchNorm1d(next_dims))
            net_arch.append(self.activation)
            curr_dims = next_dims
        
        penultimate = nn.Sequential(*net_arch).float()
        net_arch.append(nn.Linear(curr_dims, self.output_dims))
        if self.final_activation:
            net_arch.append(self.final_activation)
        if self.layer_norm:
            net_arch.append(nn.LayerNorm(self.output_dims))
            net_arch.append(nn.Tanh())
        output = nn.Sequential(*net_arch).float()
        return penultimate, output

    def _initialize(self):
        for m in self.output.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

class D4RLPolicy:
    def __init__(self, pi_filepath, noise = 0.1):
        with open(pi_filepath, 'rb') as f:
            weights = pickle.load(f)
        self.fc0_w = torch.Tensor(weights['fc0/weight'])
        self.fc0_b = torch.Tensor(weights['fc0/bias']).reshape(-1, 1)
        self.fc1_w = torch.Tensor(weights['fc1/weight'])
        self.fc1_b = torch.Tensor(weights['fc1/bias']).reshape(-1, 1)
        self.fclast_w = torch.Tensor(weights['last_fc/weight'])
        self.fclast_b = torch.Tensor(weights['last_fc/bias']).reshape(-1, 1)
        self.fclast_w_logstd = torch.Tensor(weights['last_fc_log_std/weight'])
        self.fclast_b_logstd = torch.Tensor(weights['last_fc_log_std/bias']).reshape(-1, 1)
        #lambda x: torch.maximum(x, 0)
        self.nonlinearity = nn.Tanh() if weights['nonlinearity'] == 'tanh' else nn.ReLU()

        identity = lambda x: x
        self.output_transformation = nn.Tanh() if weights[
            'output_distribution'] == 'tanh_gaussian' else nn.Identity()
        self.noise = noise

    def _get_action(self, state):
        state = torch.Tensor(state)
        single_state = False
        if len(state.shape) == 1:
            state = state.reshape((1, -1))
            single_state = True

        x = torch.matmul(state, self.fc0_w.T)
        bias = self.fc0_b.expand(self.fc0_b.shape[0], x.shape[0]).T
        x = x + bias
        x = self.nonlinearity(x)
        x = torch.matmul(x, self.fc1_w.T)
        bias = self.fc1_b.expand(self.fc1_b.shape[0], x.shape[0]).T
        x = x + bias
        x = self.nonlinearity(x)

        mean = torch.matmul(x, self.fclast_w.T)
        bias = self.fclast_b.expand(self.fclast_b.shape[0], mean.shape[0]).T
        mean = mean + bias

        logstd = torch.matmul(x, self.fclast_w_logstd.T)
        bias = self.fclast_b_logstd.expand(self.fclast_b_logstd.shape[0], logstd.shape[0]).T
        logstd = logstd + bias

        action = self.output_transformation(mean + torch.exp(logstd) * self.noise)#.T
        if single_state:
            action = action.reshape(-1)
        return action.detach().numpy()

    def __call__(self, state):
        action = self._get_action(state)
        return action
    
    def batch_sample(self, states):
        actions = self._get_action(states)
        return actions

class RandomGymPolicy:
    def __init__(self, env):
        self.env = env
        self.num_act = env.action_space.n

    def __call__(self, s, stochastic = True):
        a = self.env.action_space.sample()
        return a

    def get_action_dist(self, states):
        num_states = states.shape[0]
        act_dist = np.array([[1. / self.num_act for _ in range(self.num_act)] for _ in range(num_states)])
        return act_dist

class StablebaselinePolicy:
    def __init__(self, name, algo, env, gamma = 0.99,\
        learning_rate = 3e-4, pretrained_path = None,\
        eps_greed = 0, verbose = 0, deterministic = False):
        self.name = name
        self.env = env
        self.eps_greed = eps_greed
        #self.eq_prob = np.array([1. / self.env.action_dims for _ in range(self.env.action_dims)])
        self.deterministic = deterministic
        self.pretrained_path = pretrained_path
        if algo == 'SAC':
            self.pi = SAC.load(pretrained_path)

    def learn(self, total_timesteps, callback = None):
        self.pi.learn(total_timesteps = total_timesteps, callback = callback)

    def save(self, fname):
        self.pi.save(fname)

    def __call__(self, state):
        # act_random = np.random.binomial(n = 1, p = self.eps_greed)
        # if act_random:
        #     return self.env.action_space.sample()
        # else:
        return self.pi.predict(state, deterministic = self.deterministic)[0]
    
    def get_prob(self, state, action):

        act_dists = self.get_action_dist(state)
        action = action.reshape(-1)
        return act_dists[np.arange(len(act_dists)), action]

        '''
        act = self.pi.predict(state, deterministic = True)[0]
        if act == action:
            return (1. - self.eps_greed) * 1. + self.eps_greed * (1. / self.env.action_dims)
        else:
            return self.eps_greed * (1. / self.env.action_dims)
        '''
    
    def get_action_dist(self, states):
        dists = self.eps_greed * np.array([self.eq_prob for _ in range(len(states))])
        greedy_acts = self.pi.predict(states, deterministic = True)[0]
        dists[np.arange(len(dists)), greedy_acts] += (1. - self.eps_greed)
        return dists

    def batch_sample(self, states):
        actions = self.pi.predict(states, deterministic = self.deterministic)[0]
        # act_dists = self.get_action_dist(states)
        # actions = (act_dists.cumsum(axis = 1) > np.random.rand(act_dists.shape[0])[:, None]).argmax(axis = 1)
        # return actions
        return actions
