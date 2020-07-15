import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class Edgebase:
    """
    Base class for edges in distributed learning.
    """
    def __init__(self, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, eta=0, eta0 = 0, L = 0, local_epochs = 0):
        # from fedprox
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        
        if(batch_size == 0):
            self.batch_size = len(train_data)
        else:
            self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.eta = eta
        self.eta0 = eta0,
        self.L = L
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for FEDL.
        self.local_optimal = copy.deepcopy(list(self.model.parameters()))
        self.dt = copy.deepcopy(list(self.model.parameters()))
        self.server_grad    = copy.deepcopy(list(self.model.parameters()))
        self.pre_local_grad = copy.deepcopy(list(self.model.parameters()))

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    #def set_server_parameters(self, model):
    #    for new_param, server_param in zip(model.parameters(), self.server_parameters):
    #        server_param.data = new_param.data.clone()
            
    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def get_dt(self):
        for param in self.dt:
            param.detach()
        return self.dt

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
                            
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()
            param.grad.data = new_param.grad.data.clone()

    def get_grads(self, grads):

        self.optimizer.zero_grad()
        
        for x, y in self.trainloaderfull:
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
        self.clone_model_paramenter(self.model.parameters(), grads)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return test_acc, y.shape[0]

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss , self.train_samples
    
    def update_direction(self):
        #self.model.parameters() is model updated from server
        for opitmal_param, server_param, dt in zip(self.local_optimal,self.model.parameters(), self.dt):
            dt.data = 1/self.eta * (opitmal_param.data - server_param.data)

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X, y)
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "edge_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
