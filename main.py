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

def main(dataset, algorithm, model, batch_size, learning_rate, eta, eta0, L, num_glob_iters,
         local_epochs, optimizer, numedges, times):

    for i in range(times):
        print("---------------Running time:------------",i)

        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic(), model
            elif( dataset == "Fenist"):
                model = Mclr_Logistic(), model

        if(model == "linear_regression"):
            model = Linear_Regression(40,1), model

        if model == "logistic_regression":
            model = Logistic_Regression(40), model
        # select algorithm

        server = Server(dataset, algorithm, model, batch_size, learning_rate, eta, eta0,  L, num_glob_iters, local_epochs, optimizer, numedges, i)

        server.train()
        server.test()

    # Average data 
    #average_data(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, eta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Logistic_synthetic", choices=["Mnist","Fenist", "Linear_synthetic", "Logistic_synthetic"])
    parser.add_argument("--model", type=str, default="logistic_regression", choices=["linear_regression", "mclr", "cnn", "logistic_regression"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate for first order algorithm")
    parser.add_argument("--eta", type=float, default=0.01, help="first eta")
    parser.add_argument("--eta0", type=float, default=1, help="second eta")
    parser.add_argument("--L", type=int, default=0.01, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="SGD",choices=["SGD"])
    parser.add_argument("--algorithm", type=str, default="DANE",choices=["SecondOrder", "FirstOrder","DANE", "New", "FedAvg"])
    parser.add_argument("--numedges", type=int, default=4,help="Number of Edges per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("eta       : {}".format(args.eta))
    print("Subset of edges      : {}".format(args.numedges))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eta = args.eta,
        eta0 = args.eta0,
        L = args.L,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numedges=args.numedges,
        times = args.times
        )
