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

class edgeNew(Edgebase):
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
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        self.model.train()

        server_model = copy.deepcopy(list(self.model.parameters()))  # w^t

        X, y = self.get_next_train_batch()
        self.model.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        grads = torch.autograd.grad(loss, list(self.model.parameters()), create_graph=True, retain_graph=True)
        for param, grad in zip(list(self.model.parameters()), grads):
            param.data.add_(-grad)  # x^t_{i,0} = w^t - d^t_{i,0}

        for epoch in range(1, self.local_epochs + 1):  # r = 1..R
            X, y = self.get_next_train_batch()
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            grads = torch.autograd.grad(loss, list(self.model.parameters()), retain_graph=True)
            for param, grad in zip(list(self.model.parameters()), grads):
                param.data.add_(-self.alpha * grad)

        # param = x, server_param = w^t
        for dt, param, server_param in zip(self.dt, self.model.parameters(), server_model):
            dt.data = (1 / self.alpha) * (param - server_param)

        # save optimal parameter after training.
        self.clone_model_paramenter(self.model.parameters(), self.local_optimal)
        return loss








































