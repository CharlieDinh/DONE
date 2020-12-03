#!/usr/bin/env python
from comet_ml import Experiment
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

def main(experiment, dataset, algorithm, model, batch_size, learning_rate, alpha, eta, L, rho, num_glob_iters,
         local_epochs, optimizer, numedges, times, commet, gpu):

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for i in range(times):
        print("---------------Running time:------------",i)

        # Generate model
        if(model == "mclr"):
            model = Mclr_Logistic().to(device), model

        if(model == "linear_regression"):
            model = Linear_Regression(40,1).to(device), model

        if model == "logistic_regression":
            model = Logistic_Regression(40).to(device), model
        # select algorithm
        if(commet):
            experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "b_" + str(learning_rate) + "lr_" + str(alpha) + "al_" + str(eta) + "eta_" + str(L) + "L_" + str(rho) + "p_" +  str(num_glob_iters) + "ge_"+ str(local_epochs) + "le_"+ str(numedges) +"u")
        server = Server(experiment, device, dataset, algorithm, model, batch_size, learning_rate, alpha, eta,  L, num_glob_iters, local_epochs, optimizer, numedges, i)
        
        server.train()
        server.test()

    # Average data 
    #average_data(num_users=numedges, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L, learning_rate=learning_rate, alpha, algorithms=algorithm, batch_size=batch_size, dataset=dataset, times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Linear_synthetic", "Fashion_Mnist", "Cifar10"])
    parser.add_argument("--model", type=str, default="mclr", choices=["linear_regression", "mclr", "logistic_regression"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1, help="Local learning rate for DANE, GD")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for DONE and use alpha as eta of DANE")
    parser.add_argument("--eta", type=float, default=1, help="eta not use at this version")
    parser.add_argument("--L", type=int, default=0, help="Regularization term")
    parser.add_argument("--rho", type=int, default=0, help="Condition number")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD",choices=["SGD"])
    parser.add_argument("--algorithm", type=str, default="DONE",choices=["DONE", "GD", "DANE", "Newton","GT","PGT"])
    parser.add_argument("--numedges", type=int, default=32,help="Number of Edges per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to comet")
    parser.add_argument("--gpu", type=int, default=-1, help="Which GPU to run the experiments")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("alpha       : {}".format(args.alpha))
    print("Subset of edges      : {}".format(args.numedges))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    # Create an experiment with your api key:
    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-learning",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta" : args.beta, 
            "L_k" : args.L_k,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.numusers,
            "K" : args.K,
            "personal_learning_rate" : args.personal_learning_rate,
            "times" : args.times,
            "gpu": args.gpu
        }
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha = args.alpha,
        eta = args.eta,
        L = args.L,
        rho = args.rho,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numedges=args.numedges,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu
        )
