#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)
#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from algorithms.server.server import Server
from algorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
    
numedges = [32,32,32,32,32,32,32,32]
num_glob_iters = 100
dataset = "Linear_synthetic"

# defind parameters

if(1):
    numedges = [32,32,32,32,32,32,32,32]
    local_epochs = [40, 40, 40, 40, 40, 1]
    learning_rate = [1, 1, 1, 1, 0.1, 0.8]
    alpha = [0.1, 0.1, 0.1, 0.1, 1, 1]
    eta = [1, 1, 1, 1, 1,1]
    batch_size = [0, 256, 128, 0, 0, 0]
    algorithms = ["DONE", "DONE", "DONE", "Newton", "DANE", "GD"]
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_linear2(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,
                        learning_rate=learning_rate, alpha=alpha, eta=eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(1):
    numedges = [8,16,24,32]
    local_epochs = [40, 40, 40, 40]
    learning_rate = [1, 1, 1, 1]
    alpha = [ 0.008, 0.02, 0.05, 0.1]
    eta = [1, 1, 1, 1, 1,1]
    batch_size = [0, 0, 0, 0, 0, 0]
    algorithms = ["DONE", "DONE", "DONE", "DONE"]
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_linear_edge(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,
                        learning_rate=learning_rate, alpha=alpha, eta=eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(1):
    numedges = [32,32,32,32,32,32,32,32]
    local_epochs = [10, 20, 30, 40, 20,20,20,20]
    learning_rate = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = [0.1, 0.1, 0.1, 0.1, 0.06, 0.08, 0.1, 0.2]
    eta = [1, 1, 1, 1, 1, 1, 1, 1]
    batch_size = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    algorithms = ["DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_linear_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,
                        learning_rate=learning_rate, alpha=alpha, eta=eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(1):
    numedges = [32,32,32,32,32,32,32,32]
    local_epochs = [20, 20, 20, 20, 40, 40, 40, 40]
    learning_rate = [1, 1, 1, 1, 1, 1, 1, 1]
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    eta = [1, 1, 1, 1, 1, 1, 1, 1]
    batch_size = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    algorithms = ["DONE", "DONE", "DONE", "DONE",
                "DONE", "DONE", "DONE", "DONE"]
    L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    kappa = [10, 100, 1000, 10000,10, 100, 1000, 10000]
    plot_summary_linear_kappa(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,
                            learning_rate=learning_rate, alpha=alpha, eta=eta, algorithms_list=algorithms, batch_size=batch_size, kappa=kappa, dataset=dataset)
