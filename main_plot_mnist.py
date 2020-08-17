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
dataset = "MNIST"

# defind parameters
if(0):
    numedges =[32,32,32,32,32,32,32]
    local_epochs = [120,190,200,120,20,1]
    learning_rate = [1,1,1,1,0.05,0.2]
    alpha =  [0.015,0.01,0.003,0.015,1,1,1]
    eta = [1,1,1,1,1,1,1]
    batch_size = [0,256,128,0,0,0]
    algorithms = ["DONE", "DONE", "DONE", "Newton", "DANE", "GD"]
    L = [0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_mnist2(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    numedges = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    local_epochs = [20,50,80,100,100,100,100,100]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.02,0.02,0.02,0.02,0.005,0.01,0.02,0.03]
    eta = [1,1,1,1,1,1,1,1,1,1,1,1]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_mnist_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(1):
    
    numedges = [8,16,24,32]
    local_epochs = [20,50,80,120]
    learning_rate = [1,1,1,1]
    alpha =  [0.015,0.015,0.015,0.015]
    eta = [1,1,1,1]
    batch_size = [0,0,0,0]
    algorithms = ["DONE","DONE","DONE","DONE"]
    L = [0,0,0,0]
    plot_summary_mnist_edge(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
