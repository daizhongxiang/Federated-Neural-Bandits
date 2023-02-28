from NeuralTS.data_multi import Bandit_multi
from NeuralTS.learner_linear import LinearTS
from NeuralTS.learner_diag import NeuralTSDiag
from NeuralTS.learner_kernel import KernelTS

import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from scipy.optimize import minimize

import torch.nn.functional as F
import numpy as np
import pickle 

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


save_interval = 50 # save a log file after every "save_interval" iteration
max_iter = 5000 + save_interval



##### choose which baseline to runs
### algo = {"neural", "linear", "kernel"}
### style = {"ucb", "ts"}
algo = "neural"
style = "ucb"



depth = 1
width = 20
lam, nu = 0.1, 0.1

diag = False


#### choose which synthetic experiment to run: {"cosine", "square", "shuttle", "MagicTelescope"}
dataset = "shuttle"

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


stop_training_after_iter = 2000
local_training_iter = 30


run_list = np.arange(0, 3)

for itr in run_list:
    if algo == "linear":
        log_file_name = "results_fn_ucb/regret_iter_" + str(itr) + "_dataset_" + dataset + \
                    "_lam_" + str(lam) + "_nu_" + str(nu) + "_linear.pkl"
    elif algo == "kernel":
        log_file_name = "results_fn_ucb/regret_iter_" + str(itr) + "_dataset_" + dataset + \
                    "_lam_" + str(lam) + "_nu_" + str(nu) + "_kernel.pkl"
    elif algo == "neural":
        log_file_name = "results_fn_ucb/regret_iter_" + str(itr) + "_dataset_" + dataset + \
                    "_depth_" + str(depth) + "_width_" + str(width) + "_lam_" + str(lam) + \
                    "_nu_" + str(nu) + "_train_steps_" + str(local_training_iter) + "_neural_ucb.pkl"
        if diag:
            log_file_name = log_file_name[:-4] + "_diag.pkl"

    if style == "ts":
        log_file_name = log_file_name[:-4] + "_ts.pkl"


    if algo == "linear":
        l = LinearTS(context_dim, lam, nu, style)
        setattr(l, 'delay', 1)
    elif algo == "neural":
        l = NeuralTSDiag(context_dim, lam, nu=nu, hidden=width, depth=depth, style=style, diagonalize=diag)
        setattr(l, 'delay', 1)
    elif algo == "kernel":
        l = KernelTS(context_dim, lam, nu, style)
        setattr(l, 'delay', 1)

    if dataset == "shuttle" or dataset == "MagicTelescope":
        b = Bandit_multi(dataset, is_shuffle=True, seed=0)

    regrets = []
    for t in range(max_iter):
        regrets_per_agent = []

        if dataset == "shuttle" or dataset == "MagicTelescope":
            context, rwd = b.step()
            fs = rwd
        elif dataset == "cosine" or dataset == "square":
            context, rwd, fs = bandit_contextual(a_ground, K_arms)

        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]

        if t < stop_training_after_iter:
            l.train(context[arm_select], r, local_training_iter)

        reg = np.max(fs) - r

        regrets_per_agent.append(reg)

        print("iter {0} --- reward: {1} --- itr: {2}".format(t, r, itr))

        regrets.append(regrets_per_agent)
        if t % save_interval == 0:
            pickle.dump(regrets, open(log_file_name, "wb"))

