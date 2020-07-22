import torch
import os

from algorithms.edges.edgeSeOrder import edgeSeOrder
from algorithms.edges.edgeFiOrder import edgeFiOrder
from algorithms.edges.edgeDANE import edgeDANE
from algorithms.edges.edgeNew import edgeNew
from algorithms.edges.edgeAvg import edgeAvg

from algorithms.server.serverbase import ServerBase
from utils.model_utils import read_data, read_edge_data
import numpy as np

# Implementation for Central Server
class Server(ServerBase):
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, eta, eta0, L, num_glob_iters,
                 local_epochs, optimizer, num_edges, times):
        super().__init__(dataset,algorithm, model[0], batch_size, learning_rate, eta, eta0, L, num_glob_iters,
                         local_epochs, optimizer, num_edges, times)

        # Initialize data for all  edges
        data = read_data(dataset, read_optimal_weights=dataset.endswith("synthetic"))

        self.optimal_weights = None
        self.optimal_loss_unreg = None  # Unregularized loss
        self.optimal_loss_reg = None    # Regularized loss with parameter L
        if data[-1] is not None:
            # Synthetic dataset: save the optimal weights for comparison later
            self.optimal_weights = data[-2]
            self.optimal_loss_unreg = data[-1]
            self.optimal_loss_reg = (self.L / 2) * (np.linalg.norm(data[-1]) ** 2)

        total_edges = len(data[0])

        for i in range(total_edges):
            id, train, test = read_edge_data(i, data, dataset)

            if(algorithm == "SecondOrder"):
                edge = edgeSeOrder(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating SecondOrder server.")
            if(algorithm == "FirstOrder"):
                edge = edgeFiOrder(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating FirstOrder server.")
            if(algorithm == "DANE"):
                edge = edgeDANE(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating DANE server.")
            if algorithm == "New":
                edge = edgeNew(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
            if algorithm == "FedAvg":
                edge = edgeAvg(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)

            self.edges.append(edge)
            self.total_train_samples += edge.train_samples
            
        print("Number of edges / total edges:", num_edges, " / ", total_edges)
        
    def send_grads(self):
        assert (self.edges is not None and len(self.edges) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for edge in self.edges:
            edge.set_grads(grads)

    def train(self):
        loss = []
        if(self.algorithm == "FirstOrder"):
            # All edge will eun GD or SGD to obtain w*
            for edge in self.edges:
                edge.train(self.local_epochs)
            
            # Communication rounds
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate() # still evaluate on the global model
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.update_direction()
                    
                self.aggregate_parameters()

        elif self.algorithm == "DANE":
        
            # Choose all edges in the training process
            self.selected_edges = self.edges
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.aggregate_grads()
                self.send_grads()

                for edge in self.selected_edges:
                    edge.train(self.local_epochs)

                self.aggregate_parameters()
                self.send_parameters()
                self.evaluate()

        elif self.algorithm == "New":
            for glob_iter in range(1, self.num_glob_iters):
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()

                for edge in self.selected_edges:
                    edge.train(self.local_epochs)
                self.aggregate_parameters()
            self.save_results()
            self.save_model()

        elif self.algorithm == "FedAvg":
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)
                self.aggregate_parameters()
                
        elif self.algorithm == "SecondOrder": # Second Order method
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")

                # recive parameter from server
                self.send_parameters()
                self.evaluate()
                #self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                # Caculate gradient to send to server for average
                for edge in self.edges:
                    edge.get_full_grad()
                
                self.aggregate_grads()
                # receive average gradient form server 
                self.send_grads()
               
                # all note are trained 
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)

                self.aggregate_parameters()

        self.save_results()
        self.save_model()

    def weights_difference(self, weights=None, optimal_weights=None):
        """
        Calculate the norm of w - w*, the difference between the current weights
        and the optimal weights of the dataset.
        """
        if weights is None:
            weights = list(self.model.parameters())[0].data.clone().detach().flatten().numpy()
        if optimal_weights is None:
            optimal_weights = self.optimal_weights
        if weights.shape != optimal_weights.shape:
            weights = weights.T
        return np.linalg.norm(weights - optimal_weights)

    def regularize(self, model=None):
        model = self.model if model is None else model
        reg = 0
        for param in model.parameters():
            if param.requires_grad:
                reg += param.norm() ** 2
        return (self.L / 2) * reg

    def losses_difference(self, loss, optimal_loss=None, regularize=True):
        """
        Calculate f(w) - f(w*), the difference between the evaluation function
        at the current weights and at the optimal weights.
        """
        if optimal_loss is None:
            if regularize:
                optimal_loss = self.optimal_loss_reg
            else:
                optimal_loss = self.optimal_loss_unreg
        return loss - optimal_loss

