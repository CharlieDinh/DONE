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
    
  
numedges = 10
num_glob_iters = 600
dataset = "Mnist"
local_epochs = 20
learning_rate = 0.001
eta =  0.5
eta0 = 0.5
batch_size = 32
algorithm = "FirstOrder"
optimizer = "SGD"
i = 0
model =  "mclr"
model = Mclr_Logistic(), model
L = 0

if(1):
    server = Server(dataset, algorithm, model, batch_size, learning_rate, eta, eta0, L, num_glob_iters, local_epochs, optimizer, numedges, i)
    server.train()
    server.test()

if(0):
    local_ep = [20,20]
    L = [0,0]
    learning_rate = [0.2, 0.001]
    hyper_learning_rate =  [0, 0, 0]
    batch_size = [32,32,32]
    algorithms = ["Neumann","FedAvg"]
    plot_summary_one_figure(num_users=numedges, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, hyper_learning_rate = hyper_learning_rate, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
