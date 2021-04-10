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

# Implementation for Conjugate gradient method clients

class edgeGIANT(Edgebase):
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
        
        
        tol = 1e-8 # threshold for stopping cg iterations
        r_vec =[]
        p_vec =[]
        for param, d, grad in zip(self.model.parameters(), self.dt,self.server_grad):
            # Set direction to 0 at the begining
            #d.data = - 0 * param.grad.data.clone()
            grad = grad.grad.detach().clone()
            add_params(grad,d.data,ratio=self.alpha)
            r_vec.append(-grad)
            p_vec.append(-grad)
        
        # conjugate gradien initials 
        rsold = dot_product(r_vec, r_vec)
        
        # conjugate gradien iteration
        for i in range(1, self.local_epochs + 1):  # R
            hess_p = self.hessian_vec_prod(loss, list(self.model.parameters()),p_vec)
            add_params(hess_p,p_vec,ratio=self.alpha)

            alpha  = rsold /dot_product(p_vec,hess_p)
            
            add_params(self.dt,p_vec,ratio=alpha)
            add_params(r_vec,hess_p,ratio=-alpha)
            
            rsnew = dot_product(r_vec, r_vec)
            if np.sqrt(rsnew.cpu().detach().numpy()) < tol:
                #print('Itr:', i)
                break
            else:
                p_vec = add_params(r_vec,p_vec,ratio=(rsnew / rsold),in_place=False)
                rsold = rsnew

    def hessian_vec_prod(self, loss, params, dt):
        self.model.zero_grad()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        self.model.zero_grad()
        hv = torch.autograd.grad(grads, params, dt, only_inputs=True, create_graph=True, retain_graph=True)
        return hv
          