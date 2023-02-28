from NeuralTS.learner_diag import NeuralTSDiag
from NeuralTS.data_multi import Bandit_multi


import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import torch.nn.functional as F

import numpy as np

import pickle 

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


save_interval = 50 # save a log file after every "save_interval" iteration
max_iter = 5000 + save_interval


midpoint1 = 3
midpoint2 = 700
growing_seq = np.arange(midpoint2) / midpoint2
alpha_ts = np.append(np.zeros(midpoint1), growing_seq)
alpha_ts = np.append(alpha_ts, np.ones(max_iter - len(alpha_ts) + 5))


depth = 1
width = 20

lam, nu = 0.1, 0.1
nu_2 = 0.1


N = 2 # number of agents

diag = False # whether to use diagonalization
stop_training_after_iter = 2000


flag_not_Less_Comm = False # Set as False by default; if don't want to run FN-UCB (Less Comm.), set this flag to True


#### choose which synthetic experiment to run: {"cosine", "square", "shuttle", "MagicTelescope"}
dataset = "cosine"


if dataset == "cosine" or dataset == "square":
    # define the contextual bandit function for synthetic functions
    K_arms = 4
    context_dim = 10
    def bandit_contextual(a_ground, K_arms):
        context = []
        rwd = []
        fs = []
        for k in range(K_arms):
            x = np.random.random(context_dim) - 0.5
            x = x / np.sqrt(np.sum(x * x))
            context.append(x)
            if dataset == "cosine":
                f = np.cos(3 * np.sum(a_ground * x))
            elif dataset == "square":
                f = 10 * np.sum(a_ground * x)**2
            y = f + np.random.normal(scale=0.01)
            rwd.append(y)
            fs.append(f)
        context = np.array(context)
        rwd = np.array(rwd)
        return context, rwd, fs
    a_ground = pickle.load(open("a_groundtruth_synth.pkl", "rb"))

elif dataset == "shuttle" or dataset == "MagicTelescope":
    b = Bandit_multi(dataset, is_shuffle=True, seed=0)
    context_dim = b.dim





local_training_iter = 30


D = 0

run_list = np.arange(0, 3)


for itr in run_list:
    log_file_name = "results_fn_ucb/regret_iter_" + str(itr) + "_dataset_" + dataset + \
                "_N_" + str(N) + \
                "_depth_" + str(depth) + "_width_" + str(width) + "_lam_" + str(lam) + \
                "_nu_" + str(nu) + "_nu2_" + str(nu_2) + \
                "_train_steps_" + str(local_training_iter) + "_D_" + str(D) + ".pkl"
    if diag:
        log_file_name = log_file_name[:-4] + "_diag.pkl"

        
    func_0 = extend(Network(context_dim, hidden_size=width, depth=depth)) # this is the global NN with theta_0
    theta_0 = [p for p in func_0.parameters()]

    all_p = []
    for param in theta_0:
        all_p += list(param.detach().numpy().ravel())
    all_p = np.array(all_p)
    p = all_p.shape[0] # total number of paramters, i.e., the dimension of neural tangent features

    
    l_list = []
    b_list = []
    for i in range(N):
        l = NeuralTSDiag(context_dim, lam, nu=nu, hidden=width, depth=depth, style="ucb", init_params=theta_0)
        delay = 1
        setattr(l, 'delay', delay)
        l_list.append(l)
        
        if dataset == "shuttle" or dataset == "MagicTelescope":
            b = Bandit_multi(dataset, is_shuffle=True, seed=i)
            b_list.append(b)

    
    W_new_list = []
    B_new_list = []
    V_local_list = []
    for i in range(N):
        if diag:
            W_new_list.append(torch.zeros(p))
            V_local_list.append(lam * torch.ones(p))
        else:
            W_new_list.append(torch.zeros(p, p))
            V_local_list.append(lam * torch.diag(torch.ones(p)))
        B_new_list.append(torch.zeros(p))

    if diag:
        W_sync = torch.zeros(p)
    else:
        W_sync = torch.zeros(p, p)
    B_sync = torch.zeros(p)
    
    regrets = []
    state_dict_list = [[] for i in range(N)]

    t_last = 0
    if diag:
        V_last = lam * torch.ones(p)
    else:
        V_last = lam * torch.diag(torch.ones(p))
        
    V_t_i_bar = lam * torch.diag(torch.ones(p))
    theta_t_i_bar = torch.zeros(p)

    
    communication_flag = np.zeros(N)
    communicated_last_round = False

    
    func_agg = extend(Network(context_dim, hidden_size=width, depth=depth, init_params=theta_0))
    V_sync_NN_inv = torch.inverse(lam * torch.diag(torch.ones(p)))

    
    all_communication_flags = []
    sdFinal = None
    for t in range(max_iter):
        regrets_per_agent = []
        if np.any(communication_flag):
            communicated_last_round = True
        context_list = []
        arm_select_list = []
        r_list = []

        for i in range(N):
            
            if dataset == "shuttle" or dataset == "MagicTelescope":
                context, rwd = b_list[i].step()
                fs = rwd
            elif dataset == "cosine" or dataset == "square":
                context, rwd, fs = bandit_contextual(a_ground, K_arms)

            
            if communicated_last_round:
                if diag:
                    V_last = lam * torch.ones(p) + W_sync
                else:
                    V_last = lam * torch.diag(torch.ones(p)) + W_sync
                    
                t_last = t - 1

                if diag:
                    W_new_list[i] = torch.zeros(p)
                else:
                    W_new_list[i] = torch.zeros(p, p)
                B_new_list[i] = torch.zeros(p)

                communication_flag[i] = 0


            if diag:
                V_t_i_bar = lam * torch.ones(p) + W_sync + W_new_list[i]
                theta_t_i_bar = (B_sync + B_new_list[i]) / V_t_i_bar
            else:
                V_t_i_bar = lam * torch.diag(torch.ones(p)) + W_sync + W_new_list[i]
                theta_t_i_bar =  torch.matmul(torch.inverse(V_t_i_bar), B_sync + B_new_list[i])
    

            tensor = torch.from_numpy(context).float()
            mu = func_0(tensor)
            sum_mu = torch.sum(mu)
            with backpack(BatchGrad()):
                sum_mu.backward()
            g_0_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in func_0.parameters()], dim=1)


            UCB_2_first = torch.inner(g_0_list, theta_t_i_bar)

            if diag:
                UCB_2_second = nu * np.sqrt(lam) * torch.sqrt(torch.sum(g_0_list * g_0_list / V_t_i_bar, dim=1))
            else:
                tmp = torch.matmul(g_0_list, torch.inverse(V_t_i_bar))
                UCB_2_second = nu * np.sqrt(lam) * torch.sqrt(torch.matmul(tmp, torch.transpose(g_0_list, 0, 1)))
                UCB_2_second = torch.diagonal(UCB_2_second, 0)

            UCB_2 = UCB_2_first + UCB_2_second

            
            if alpha_ts[t]>0:
                UCB_1_first = func_agg(tensor)
                UCB_1_first = torch.squeeze(UCB_1_first)

                if diag:
                    if not flag_not_Less_Comm:
                        UCB_1_second = nu_2 * np.sqrt(lam) * torch.sqrt(torch.sum(g_0_list * g_0_list * V_sync_NN_inv, dim=1))
                    else:
                        UCB_1_second = torch.zeros(len(context))
                        for ii in range(N):
                            UCB_1_second += (1/N) * nu_2 * np.sqrt(lam) * torch.sqrt(torch.sum(g_0_list * g_0_list * (1 / V_local_list[ii]), dim=1))

                else:
                    if not flag_not_Less_Comm:
                        tmp = torch.matmul(g_0_list, V_sync_NN_inv)
                        UCB_1_second = nu_2 * np.sqrt(lam) * torch.sqrt(torch.matmul(tmp, torch.transpose(g_0_list, 0, 1)))
                        UCB_1_second = torch.diagonal(UCB_1_second, 0)
                    else:
                        UCB_1_second = torch.zeros(len(context))
                        for ii in range(N):
                            tmp = torch.matmul(g_0_list, torch.inverse(V_local_list[ii]))
                            UCB_1_second_tmp = nu_2 * np.sqrt(lam) * torch.sqrt(torch.matmul(tmp, torch.transpose(g_0_list, 0, 1)))
                            UCB_1_second += (1/N) * torch.diagonal(UCB_1_second_tmp, 0)




                UCB_1_first = UCB_1_first.view(-1)
                UCB_1 = UCB_1_first + UCB_1_second

                UCB_2 = alpha_ts[t] * UCB_1 + (1 - alpha_ts[t]) * UCB_2

            
            arm_select = torch.argmax(UCB_2)
            r = rwd[arm_select]
            
            if diag:
                W_new_list[i] += g_0_list[arm_select] * g_0_list[arm_select]
            else:
                W_new_list[i] += torch.outer(g_0_list[arm_select], g_0_list[arm_select])

            B_new_list[i] += r * g_0_list[arm_select]

            if diag:
                V_local_list[i] += g_0_list[arm_select] * g_0_list[arm_select]
            else:
                V_local_list[i] += torch.outer(g_0_list[arm_select], g_0_list[arm_select])
            if diag:
                V_t_i = lam * torch.ones(p) + W_sync + W_new_list[i]
            else:
                V_t_i = lam * torch.diag(torch.ones(p)) + W_sync + W_new_list[i]


            if not diag:
                criterion = torch.sum(torch.log(torch.diagonal(V_t_i, 0))) - \
                        torch.sum(torch.log(torch.diagonal(V_last, 0)))
            else:
                criterion = torch.sum(torch.log(V_t_i)) - torch.sum(torch.log(V_last))
            if (t - t_last) * criterion > D:
                communication_flag[i] = 1

            reg = np.max(fs) - r

            regrets_per_agent.append(reg)

            context_list.append(context)
            arm_select_list.append(arm_select)
            r_list.append(r)

            
            print("iter {0} --- agent {1} --- reward: {2} --- itr: {3}".format(t, i, r, itr))

        communicated_last_round = False

        if np.any(communication_flag):
            if t < stop_training_after_iter and alpha_ts[t+1]>0:
                for i in range(N):
                    state_dict = l_list[i].train(context_list[i][arm_select_list[i]], r_list[i], local_training_iter, \
                                                init_state_dict=None)
                    state_dict_list[i] = state_dict


        all_communication_flags.append(np.any(communication_flag))
        
        regrets.append(regrets_per_agent)
        if t % save_interval == 0:
            all_info = {"regrets":regrets, "communication_flag":all_communication_flags}
            pickle.dump(all_info, open(log_file_name, "wb"))


        
        ## below is done by the central server
        if np.any(communication_flag):
            for i in range(N):
                W_sync += W_new_list[i]
                B_sync += B_new_list[i]

            if alpha_ts[t+1] > 0:
                if diag:
                    V_sync_NN_inv = torch.zeros(p)
                else:
                    V_sync_NN_inv = torch.zeros(p, p)
                for i in range(N):
                    if diag:
                        V_sync_NN_inv += (1 / V_local_list[i]) / N
                    else:
                        V_sync_NN_inv += torch.inverse(V_local_list[i]) / N

                if t<stop_training_after_iter:
                    ##### NN parameter aggregation
                    sdFinal = state_dict_list[0]
                    # Average all parameters
                    for key in state_dict_list[0]:
                        test = torch.zeros(state_dict_list[0][key].shape)
                        for i in range(N):
                            test += state_dict_list[i][key] / N
                        sdFinal[key] = test
                    func_agg = extend(Network(context_dim, hidden_size=width, depth=depth))
                    func_agg.load_state_dict(sdFinal)
