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
    
numedges = 64
num_glob_iters = 200
dataset = "Mnist"
optimizer = "SGD"
model =  "mclr"
times = 1
model = Mclr_Logistic(), model

# defind parameters
local_epochs = [20,20,20]
learning_rate = [1,1,1]
eta =  [0.05,0.05,1]
eta0 = [0.1,1,1]
batch_size = [0,0,0]
algorithms = ["SecondOrder","SecondOrder","DANE"]
L = [0,0,0]

plot_summary_one_figure(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, eta = eta, eta0 = eta0, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset)
