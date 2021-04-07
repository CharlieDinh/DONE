import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *


# Implementation for FedAvg clients

class edgeDONE(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.grad = new_grad.data.clone()
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.grad = new_grads[idx].clone()

    def get_full_grad(self):
        for X, y in self.trainloaderfull:
            X, y = (X.to(self.device), y.to(self.device))
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

    def train(self, epochs, glob_iter):
        # Set initial value for d0, d0 can be 0 or can be any value
        for d, server_param, local_param in zip(self.dt, self.server_grad, self.model.parameters()):
            if(self.batch_size != 0):
                d.data =  - server_param.grad.data.clone()
            else:
                d.data =  - local_param.grad.data.clone()
            
        self.model.zero_grad()
        self.model.train()

        # Sample a mini-batch (D_i)
        (X, y) = self.get_next_train_batch()
        loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
        # loss.backward(create_graph=True)
        #grads = []
        # Richardson iteration
        for i in range(1, self.local_epochs + 1):  # R
            hess_vec_prods = self.hessian_vec_prod(loss, list(self.model.parameters()), self.dt)
            for grad, d, hess_vec in zip(self.server_grad, self.dt, hess_vec_prods):
                # d.data = d.data - tempalpha * hess_vec - grad.data
                d.data = d.data - self.alpha * hess_vec - self.alpha * grad.grad.data

    def hessian_vec_prod(self, loss, params, dt):
        self.model.zero_grad()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        self.model.zero_grad()
        hv = torch.autograd.grad(grads, params, dt, only_inputs=True, create_graph=True, retain_graph=True)
        return hv

