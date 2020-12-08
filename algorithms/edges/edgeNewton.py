import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from algorithms.edges.nn_utils import hessian


# Implementation for FedAvg clients

class edgeNewton(Edgebase):
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
        self.hessian = None

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.grad = new_grad.data.clone()
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.grad = new_grads[idx].clone()

    def set_dt(self, new_dt):
        for idx, dt in enumerate(self.dt):
            dt.data = new_dt[idx]

    def get_full_grad(self):
        for X, y in self.trainloaderfull:
            X, y = (X.to(self.device), y.to(self.device))
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

    def gethessianproduct(self, epochs, glob_iter):
        self.model.zero_grad()

        for X, y in self.trainloaderfull:
            X,y = (X.to(self.device),y.to(self.device))
            loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
            loss.backward(create_graph=True)
            self.Hdt = self.hessian_vec_prod(loss, list(self.model.parameters()), self.dt)
            
    def hessian_vec_prod(self, loss, params, dt):
        self.model.zero_grad()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        self.model.zero_grad()
        hv = torch.autograd.grad(grads, params, dt, only_inputs=True, create_graph=True, retain_graph=True)
        return hv

    def get_hessian(self):
        for X, y in self.trainloaderfull:
            loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
            loss.backward(create_graph=True)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, retain_graph=True)
            self.hessian = hessian(grads, self.model)

    def send_hessian(self):
        self.get_hessian()
        return self.hessian.clone().detach()