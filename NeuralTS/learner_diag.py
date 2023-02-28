import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from scipy.optimize import minimize

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
        if init_params is None:
            ## use initialization
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            ### manually set the initialization vector
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y


class NeuralTSDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, depth=1, style='ts', init_x=None, init_y=None, init_params=None, diagonalize=True):

        self.diagonalize = diagonalize
        self.func = extend(Network(dim, hidden_size=hidden, depth=depth, init_params=init_params))

        if init_x is not None:
            self.context_list = torch.from_numpy(init_x).to(dtype=torch.float32)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = torch.from_numpy(init_y).to(dtype=torch.float32)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            ### diagonalization
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            ### no diagonalization
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()


    def select(self, context):
        tensor = torch.from_numpy(context).float()
        mu = self.func(tensor)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()

        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)

        if self.diagonalize:
#             ### diagonalization
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        else:
            ### no diagonalization
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(self.nu * self.lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if self.style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if self.diagonalize:
            ### diagonalization
            self.U += g_list[arm] * g_list[arm]
        else:
            ### no diagonalization
            self.U += torch.outer(g_list[arm], g_list[arm])

        return arm, g_list[arm].norm().item(), 0, 0


    def train(self, context, reward, local_training_iter=30, init_state_dict=None):
        if init_state_dict is not None:
            self.func.load_state_dict(init_state_dict)

        self.len += 1
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(dtype=torch.float32)
                self.reward = torch.tensor([reward], dtype=torch.float32)
            else:
                self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(dtype=torch.float32)))
                
                self.reward = torch.cat((self.reward, torch.tensor([reward], dtype=torch.float32)))
        if self.len % self.delay != 0:
            return 0
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred = self.func(self.context_list).view(-1)
        
            loss = self.loss_func(pred, self.reward)
            loss.backward()
            optimizer.step()
        
        return self.func.state_dict()
