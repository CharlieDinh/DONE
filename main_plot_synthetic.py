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

numedges = [32,32,32,32,32,32,32,32,32,32,32,32]
num_glob_iters = 100
dataset = "Linear_synthetic"
if(1):
    local_epochs = [5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20]
    learning_rate = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    alpha = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    algorithms = ["DONE", "DONE", "DONE", "DONE",
                "DONE", "DONE", "DONE", "DONE",
                "DONE", "DONE", "DONE", "DONE"]
    L = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    kappa = [10, 100, 1000, 10000, 10, 100, 1000, 10000, 10, 100, 1000, 10000]
    plot_summary_linear_kappa(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L,
                            learning_rate=learning_rate, alpha=alpha, eta=eta, algorithms_list=algorithms, batch_size=batch_size, kappa=kappa, dataset=dataset)
