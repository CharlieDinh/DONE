import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeDANE(Edgebase):
    def __init__(self,device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, eta, L,
                         local_epochs)

        self.pre_params = []
        if (model[1] == "linear_regression"):
            self.loss = nn.MSELoss()
        elif model[1] == "logistic_regression":
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = DANEOptimizer(self.model.parameters(), lr=self.learning_rate, L = self.L , eta = self.eta)

    # def set_grads(self, new_grads):
    #     if isinstance(new_grads, nn.Parameter):
    #         for model_grad, new_grad in zip(self.server_grad, new_grads):
    #             model_grad.data.grad = new_grad.data
    #     elif isinstance(new_grads, list):
    #         for idx, model_grad in enumerate(self.server_grad):
    #             model_grad.data.grad = new_grads[idx]

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.grad = new_grad.data.clone()
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.grad = new_grads[idx].clone()
    
    def get_full_grad(self):
        for X, y in self.trainloaderfull:
            X, y = X.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

        self.pre_local_grad = []
        self.pre_params = []

        # Find derivative of phi(w^(t-1))
        for X, y in self.trainloaderfull:
            X, y = X.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            #loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
            loss.backward()

        for param in self.model.parameters():
            self.pre_params.append(param.data.clone())
            self.pre_local_grad.append(param.grad.data.clone())


    # def train(self, epochs):
    #     self.model.train()
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         #loss_per_epoch = 0
    #         X, y = self.get_next_train_batch()
    #         #for X, y in self.trainloaderfull:
    #         self.optimizer.zero_grad()
    #         # output = self.model(X)
    #         loss = self.total_loss(X=X, y=y, regularize=True)
    #         # loss = self.loss(output, y)
    #         #loss = self.total_loss(X=X, y=y, full_batch=False, regularize=True)
    #         loss.backward()
    #         self.optimizer.step(server_grads=self.server_grad, pre_grads=self.pre_local_grad, pre_params=self.pre_params)
    #         # for batch_idx, (X, y) in enumerate(self.trainloader):
    #         #     self.optimizer.zero_grad()
    #         #     output = self.model(X)
    #         #     loss = self.loss(output, y)
    #         #     loss.backward()
    #         #     self.optimizer.step(server_grads=self.server_grad, pre_grads=self.pre_local_grad,
    #         #                         pre_params=self.pre_params)
    def train(self, epochs):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss =  self.loss(output, y)
                loss.backward()
                self.optimizer.step(server_grads=self.server_grad, pre_grads=self.pre_local_grad, pre_params=self.pre_params)