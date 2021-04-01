import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *
from algorithms.edges.nn_utils import dot_product,add_params

# Implementation for NEWTON-LEARN (λ > 0 case) method clients

class edgeNL1(Edgebase):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.second_derivative = []
        
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.grad = new_grad.data.clone()
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.grad = new_grads[idx].clone()

    def get_full_grad_2nd_dloss(self):
        self.second_derivative = []
        for X, y in self.trainloaderfull:
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            h_x_j = self.second_derivative_loss(loss, output)
            self.second_derivative.append(h_x_j)
            loss.backward()
            
        
    def second_derivative_loss(self, loss, output):
        self.model.zero_grad()
        first_derivative = torch.autograd.grad(loss,output, create_graph=True)[0]
        #self.model.zero_grad()
        # We now have dloss/dy
        second_derivative = torch.autograd.grad(first_derivative, output, grad_outputs=torch.ones_like(first_derivative))[0]
        return second_derivative        
          