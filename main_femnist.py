#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from algorithms.centralServer.Server import Server
from algorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
    
  
numusers = 10
num_glob_iters = 600
dataset = "Femnist"
local_ep = 20
learning_rate = 0.003
hyper_learning_rate =  0
batch_size = 32
algorithm = "Neumann"
optimizer = "SGD"
i = 0
model =  "mclr"
model = Mclr_Logistic(input_dim = 784, output_dim = 62), model
L = 0

if(0):

    server = Server(dataset, algorithm, model, batch_size, learning_rate, hyper_learning_rate, L, num_glob_iters, local_ep, optimizer, numusers, i)
    
    server.train()
    server.test()

    average_data(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L,learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms=algorithm, batch_size=batch_size, dataset=dataset,times = i)

if(1):
    local_ep = [20,20]
    L = [0,0]
    learning_rate = [0.003, 0.003]
    hyper_learning_rate =  [0, 0, 0]
    batch_size = [32,32,32]
    algorithms = ["Neumann","FedAvg"]
    plot_summary_one_figure(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
