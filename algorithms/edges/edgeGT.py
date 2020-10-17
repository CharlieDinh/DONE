import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

# Implementation for Conjugate gradient method clients

class edgeGT(Edgebase):
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

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.grad = new_grad.data.clone()
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.grad = new_grads[idx].clone()

    def get_full_grad(self):
        for X, y in self.trainloaderfull:
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

    def train(self, epochs, glob_iter):
        self.model.zero_grad()

        # Sample a mini-batch (D_i)
        (X, y) = self.get_next_train_batch()
        loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
        loss.backward(create_graph=True)
        
        
        for param, d in zip(self.model.parameters(), self.dt):
            # Set direction to 0 at the begining
            d.data = - 0 * param.grad.data.clone()
        
        # matrices should be flatten
        grads = torch.cat([x.grad.data.view(-1) for x in self.server_grad]).reshape(-1,1)
        dt = torch.cat([x.data.view(-1) for x in self.dt]).reshape(-1,1)

        # conjugate gradien initials 
        tol = 1e-8 # threshold for stopping cg iterations
        r = -grads #+ torch.mm(hess,dt) # direction is 0 at the begining
        p = r.clone().detach()
        rsold = torch.dot(r.view(-1), r.view(-1))
        
        # conjugate gradien iteration
        for i in range(1, self.local_epochs + 1):  # R
            index=0
            p_vec=[]
            for d in self.dt:
                shape = d.data.shape
                p_vec.append(p[index: index+ d.data.numel()].reshape(shape))
                index = index+ d.data.numel()
            hess_vec_prods = self.hessian_vec_prod(loss, list(self.model.parameters()), p_vec)
            hess_p = torch.cat([x.data.view(-1) for x in hess_vec_prods]).reshape(-1,1)
            hess_p = hess_p + self.alpha * p
            alpha  = rsold /torch.dot(p.view(-1),hess_p.view(-1))
            dt.data = dt.data + alpha * p
            r.data = r - alpha* hess_p
            rsnew = torch.dot(r.view(-1), r.view(-1))
            if np.sqrt(rsnew.detach().numpy()) < tol:
                #print('Itr:', i)
                break
            else:
                p.data = r + (rsnew / rsold)* p
                rsold.data = rsnew
        
        # coppying rsult to self.dt
        index=0
        for d in self.dt:
            shape = d.data.shape
            d.data = dt[index: index+ d.data.numel()].reshape(shape)
            index = index+ d.data.numel()
          
    def hessian_vec_prod(self, loss, params, dt):
        self.model.zero_grad()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        self.model.zero_grad()
        hv = torch.autograd.grad(grads, params, dt, only_inputs=True, create_graph=True, retain_graph=True)
        return hv    