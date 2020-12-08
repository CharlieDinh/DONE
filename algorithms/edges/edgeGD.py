import torch
import torch.nn as nn
from algorithms.edges.edgebase import Edgebase
from algorithms.optimizers.optimizer import *

class edgeGD(Edgebase):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, eta, L,
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

        self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, epochs, glob_iter):
        self.model.train()
        # Only update once time
        for X, y in self.trainloaderfull:
            X, y = X.to(self.device), y.to(self.device)
            self.model.train()
            #loss_per_epoch = 0
            # Sample a mini-batch (D_i)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
