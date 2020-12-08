import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

# Implementation for Preconditioned Conjugate Gradient method clients

class edgePGT(Edgebase):
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

    def get_full_grad(self):
        for X, y in self.trainloaderfull:
            self.model.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
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
        
        # calculating hessian
        hess = self.calc_hessian(loss, list(self.model.parameters())) 
        
        I =  torch.eye(*hess.size())
        hess = hess + self.alpha * I # adding identify notice for regularization

        
        # preconditioned conjugate gradien initials 
        
        M = torch.diagflat(torch.diag(hess)) # Jacopi preconditioner
        #M = torch.diagflat(torch.diag(hess)) + torch.triu(hess) # preconditioner
        invM = torch.inverse(M)
        
        tol = 1e-8 # threshold for stopping cg iterations
        r = torch.mm(hess,dt) - grads
        z = torch.mm(invM,r)
        p = z.clone().detach()
        rsold = torch.dot(r.view(-1), r.view(-1))
        
        # preconditioned conjugate gradien iteration
        for i in range(1, self.local_epochs + 1):  # R
            w =  torch.mm(hess,p)
            alpha  = torch.dot(r.view(-1),z.view(-1))/torch.dot(p.view(-1),w.view(-1))
            dt = dt + alpha * p
            q = torch.dot(r.view(-1),z.view(-1)) 
            r = r -alpha * w
            rsnew = torch.dot(r.view(-1), r.view(-1))
            if np.sqrt(rsnew.detach().numpy()) < tol:
                print('Itr:', i)
                break
            else:  
                z = torch.mm(invM,r)
                beta  = torch.dot(r.reshape(-1),z.reshape(-1))/q  
                p = z + beta * p
                
        # coppying rsult to self.dt
        index=0
        for d in self.dt:
            shape = d.data.shape
            d.data = dt[index: index+ d.data.numel()].reshape(shape)
            index = index+ d.data.numel()
            
            
    def calc_hessian(self, loss, params):
        self.model.zero_grad()
        loss_grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            self.model.zero_grad()
            grad2rd = torch.autograd.grad(g_vector[idx], params, create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian            