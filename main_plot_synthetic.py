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
    
numedges = 32
num_glob_iters = 100
dataset = "Linear_synthetic"

# defind parameters
# local_epochs = [15,15,15,40,20,1,30,30,30,40,20,1]
# learning_rate = [1,1,1,0.5,0.1,0.45,1,1,1,0.5,0.1,0.45]
# eta =  [0.4,0.4,0.4,1,1,1,0.2,0.2,0.2,1,1,1]
# eta0 = [1,1,1,1,1,1,1,1,1,1,1,1]
# batch_size = [0,256,0,0,256,0,0,256,0,0,256,0]
# algorithms = ["DONE","DONE", "Newton", "DANE", "FedDANE", "FirstOrder","DONE","DONE", "Newton", "DANE", "FedDANE", "FirstOrder"]
# L = [0,0,0,0,0,0,0,0,0,0,0,0]
# kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
#plot_summary_linear(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, eta = eta, eta0 = eta0, algorithms_list=algorithms, batch_size=batch_size,kappa=kappa, dataset=dataset)

local_epochs = [20,20,20,20,5,10,20,30]
learning_rate = [1,1,1,1,1,1,1,1]
eta =  [0.01,0.03,0.05,0.07,0.05,0.05,0.05,0.05]
eta0 = [1,1,1,1,1,1,1,1,1,1,1,1]
batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
algorithms = ["DONE","DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
L = [0,0,0,0,0,0,0,0,0,0,0,0]
#kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
plot_summary_linear_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, eta = eta, eta0 = eta0, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
