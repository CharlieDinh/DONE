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
    

num_glob_iters = 100

if(0):
    dataset = "Mnist"
    numedges = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    local_epochs = [10,20,30,40,40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.03,0.03,0.03,0.03,0.005,0.01,0.02,0.03]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_mnist_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)


if(0):
    dataset = "human_activity"
    numedges = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    local_epochs = [10,20,30,40,40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.02,0.02,0.02,0.02,0.005,0.01,0.015,0.02]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    #kappa = [4,4,4,4,4,4,9,9,9,9,9,9]
    plot_summary_human_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "Nist"
    numedges = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    local_epochs = [10,20,30,40,40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.01,0.01,0.01,0.01,0.004,0.006,0.008,0.01]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE", "DONE", "DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_nist_R_and_alpha(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "Mnist"
    numedges = [32, 32, 32, 32]
    local_epochs = [120,120,120,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.01,0.01,0.01,0.03]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [32,64,128,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_mnist_batch(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)


if(0):
    dataset = "Nist"
    numedges = [32, 32, 32, 32]
    local_epochs = [80,80,80,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.005,0.005,0.005,0.01]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [32,64,128,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_nist_batch(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "human_activity"
    numedges = [30, 30, 30, 30]
    local_epochs = [80,80,80,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.01,0.01,0.01,0.02]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [32, 64,128,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_human_batch(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)


if(0):
    dataset = "Mnist"
    numedges = [13, 20, 26, 32]
    local_epochs = [40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.03,0.03,0.03,0.03]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_mnist_edge(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "Nist"
    numedges = [13, 20, 26, 32]
    local_epochs = [40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.01,0.01,0.01,0.01]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_nist_edge(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "human_activity"
    numedges = [12, 18, 24, 30]
    local_epochs = [40,40,40,40]
    learning_rate = [1,1,1,1,1,1,1,1]
    alpha =  [0.02,0.02,0.02,0.02]
    eta = [1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0]
    algorithms = ["DONE","DONE", "DONE", "DONE"]
    L = [0,0,0,0,0,0,0,0,0,0,0,0]
    plot_summary_human_edge(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "Mnist"
    numedges = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    local_epochs = [40,40,40,40,40,20]
    learning_rate = [1,1,0.04,0.04,1,0.2]
    alpha =  [0.03,0.03, 0.03,0.03,0.03, 0.03]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    L = [0.02,0.02,0.02,0.02,1.0,0.02]
    plot_summary_mnist_algorithm(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "Nist"
    numedges = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    local_epochs = [40,40,40,40,40,20]
    learning_rate = [1,1,0.02,0.02,1,0.02]
    alpha =  [0.01,0.01, 0.03,0.03,0.01, 0.03]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    L = [0.02,0.02,0.02,0.02,1.0,0.02]
    plot_summary_nist_algorithm(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)

if(0):
    dataset = "human_activity"
    numedges = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    local_epochs = [40,40,40,40,40,20]
    learning_rate = [1,1,0.05,0.05,1,0.1]
    alpha =  [0.01,0.01, 0.03,0.03,0.01, 0.03]
    eta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    batch_size = [0,0,0,0,0,0,0,0,0,0,0,0]
    algorithms = ["DONE", "NEWTON", "FEDL", "DANE", "GT", "GD"]
    L = [0.02,0.02,0.02,0.02,1.0,0.02]
    plot_summary_human_algorithm(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
