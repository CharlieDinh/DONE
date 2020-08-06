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
optimizer = "SGD"
model =  "linear_regression"
times = 1
model = Linear_Regression(40,1), model

# defind parameters
local_epochs = [20]
learning_rate = [0.01]
alpha =  [0.01]
eta = [1]
batch_size = [0]
algorithms = ["DANE"]
L = [0.01]

if(1):
    for i in range(len(algorithms)):
        for time in range(times):
            server = Server(dataset, algorithms[i], model, batch_size[i], learning_rate[i], alpha[i], eta[i], L[i], num_glob_iters, local_epochs[i], optimizer, numedges, time)
            server.train()
            server.test()
        average_data(num_users=numedges, loc_ep1=local_epochs[i], Numb_Glob_Iters=num_glob_iters, lamb=L[i], learning_rate=learning_rate[i], alpha = alpha[i], eta = eta[i], algorithms=algorithms[i], batch_size=batch_size[i], dataset=dataset, times = times)

if(0):
    plot_summary_one_figure(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha = alpha, eta = eta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
