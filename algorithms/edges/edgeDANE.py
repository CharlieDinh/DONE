import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeDANE(Edgebase):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, eta,eta0, L,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, eta, eta0, L,
                         local_epochs)

        self.pre_params = []
        if model[1] == "linear_regression":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = DANEOptimizer(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.data.grad = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.data.grad = new_grads[idx]

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

        self.pre_local_grad = []
        self.pre_params = []

        # Find derivative of phi(w^(t-1))
        for X, y in self.trainloaderfull:

            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()

        for param in self.model.parameters():
            self.pre_params.append(param.data.clone())
            self.pre_local_grad.append(param.grad.data.clone())


    def train(self, epochs):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            #loss_per_epoch = 0
            for X, y in self.trainloaderfull:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(server_grads=self.server_grad, pre_grads=self.pre_local_grad,
                                    pre_params=self.pre_params)
            # for batch_idx, (X, y) in enumerate(self.trainloader):
            #     self.optimizer.zero_grad()
            #     output = self.model(X)
            #     loss = self.loss(output, y)
            #     loss.backward()
            #     self.optimizer.step(server_grads=self.server_grad, pre_grads=self.pre_local_grad,
            #                         pre_params=self.pre_params)
