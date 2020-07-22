import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeAvg(Edgebase):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, eta,eta0, L,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, eta, eta0, L,
                         local_epochs)

        self.pre_params = []
        if model[1] == "linear_regression":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.server_grad, new_grads):
                model_grad.data.grad = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.server_grad):
                model_grad.data.grad = new_grads[idx]

    def train(self, epochs, glob_iter):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            #loss_per_epoch = 0
            for X, y in self.trainloaderfull:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
