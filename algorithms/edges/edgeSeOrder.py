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

class edgeSeOrder(Edgebase):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, eta, eta0, L,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, eta, eta0, L,
                         local_epochs)

        if(model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
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
        self.update_dt()
        # self.model.train()
        # for epoch in range(1, self.local_epochs + 1):
        #     self.model.train()
        #     #loss_per_epoch = 0
        #     for batch_idx, (X, y) in enumerate(self.trainloader):
        #         self.optimizer.zero_grad()
        #         output = self.model(X)
        #         loss = self.loss(output, y)
        #         loss.backward()
        #         self.optimizer.step()
        # return loss

    def hessian_vec_prod(self, f, params: list, dt: list):
        # f.backward(create_graph=True)
        hess_vec_prods = []
        for param, d in zip(params, dt):
            if param.grad is not None:
                param.grad.zero_()
            grad, = torch.autograd.grad(f, param, create_graph=True, retain_graph=True)
            dot = grad.mul(d.clone().detach()).sum()
            if param.grad is not None:
                param.grad.zero_()
            g2, = torch.autograd.grad(dot, param, retain_graph=True)
            hess_vec_prods.append(g2.data.clone())
            #param.grad = g2.data.clone()
        return hess_vec_prods

    def update_dt(self):
        self.model.train()
        (X, y) = self.get_next_train_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, retain_graph=True)
        grads = list(grads)

        # dt_0
        for d, grad in zip(self.dt, grads):
            d.data = grad.data.clone()

        for i in range(1, self.local_epochs + 1):  # R
            hess_vec_prods = self.hessian_vec_prod(loss, list(self.model.parameters()), self.dt)
            for grad, d, hess_vec in zip(grads, self.dt, hess_vec_prods):
                d.data = d.data - self.eta * hess_vec - grad.data




































