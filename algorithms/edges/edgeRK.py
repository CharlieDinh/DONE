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

class edgeRK(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
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
        
        q = 30
        tol = 1e-8 # threshold for stopping cg iterations
        grads = []
        dt = []
        for grad, param, d in zip(self.server_grad, self.model.parameters(), self.dt):
            # Set direction to 0 at the begining
            d.data = - 0 * param.grad.data.clone()
            grads.extend(grad.grad.data.clone().detach().reshape(-1))
            dt.extend(d.data.clone().detach().reshape(-1))
        
        grads = torch.FloatTensor(grads).reshape(-1,1)
        dt = torch.FloatTensor(dt).reshape(-1)
        
        hess = self.calc_hessian(loss, list(self.model.parameters())) 
        I = torch.eye(hess.shape[0])
        hess = hess + self.alpha * I
        
        m = hess.shape[0]
        n = hess.shape[1]
        # conjugate gradien iteration
        for i in range(1, self.local_epochs + 1):  # R
            idxs = np.random.choice(np.arange(m), size=q)
            for idx in range(q):
                while torch.linalg.norm(hess[idxs[idx], :]) == 0.0:
                    idx = np.random.choice(np.arange(m), 1)
                    idxs[q] = idx
            A_ = hess[idxs, :]
            delta =  torch.zeros(n)
            for i in range(q):
                delta += A_[i].T * (A_[i] @ dt - grads[idxs][i]) /torch.linalg.norm(A_[i])**2
            dt = dt +  (1/q) * delta
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
          