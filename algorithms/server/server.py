import torch
import os

from algorithms.edges.edgeDONE import edgeDONE
#from algorithms.edges.edgeSeOrder2 import edgeSeOrder2
from algorithms.edges.edgeFiOrder import edgeFiOrder
from algorithms.edges.edgeDANE import edgeDANE
from algorithms.edges.edgeNew import edgeNew
from algorithms.edges.edgeGD import edgeGD
from algorithms.edges.edgeFEDL import edgeFEDL
from algorithms.edges.edgeNewton import edgeNewton

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
        data = read_data(dataset, read_optimal_weights=False)

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

            if(algorithm == "DONE"):
                edge = edgeDONE(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating DONE server.")
            if(algorithm == "FirstOrder"):
                edge = edgeFiOrder(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating FirstOrder server.")
            if(algorithm == "DANE"):
                edge = edgeDANE(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                #print("Finished creating DANE server.")
            if algorithm == "New":
                edge = edgeNew(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)

            if algorithm == "GD":
                edge = edgeGD(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)

            if(algorithm == "FEDL"):
                edge = edgeFEDL(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)

            if(algorithm == "Newton"):
                edge = edgeNewton(id, train, test, model, batch_size, learning_rate, eta, eta0, L, local_epochs, optimizer)
                
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

    def send_dt(self):
        for edge in self.edges:
            edge.set_dt(self.dt)

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
        
            # # Choose all edges in the training process
            # self.selected_edges = self.edges
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
            #     self.aggregate_grads()

            #     self.send_grads()
            #     self.send_parameters()

            #     self.evaluate()

            #     for edge in self.selected_edges:
            #         edge.train(self.local_epochs)

            #     self.aggregate_parameters()
            #recive parameter from server
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
                    edge.train(self.local_epochs)

                self.aggregate_parameters()          

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

        elif self.algorithm == "GD":
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
                self.selected_edges = self.select_edges(glob_iter, self.num_edges)

                for edge in self.selected_edges:
                    edge.train(self.local_epochs, glob_iter)
                self.aggregate_parameters()
                
        elif self.algorithm == "DONE": # Second Order method
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

        elif self.algorithm == "FEDL":
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.send_grads()
                self.evaluate()

                self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                for edge in self.selected_edges:
                    edge.train(self.local_epochs)
                #self.selected_edges[0].train(self.local_epochs)
                self.aggregate_parameters()
                self.aggregate_grads()

        elif self.algorithm == "Newton":
            # create zero dt
        
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()
            
                # reset all direction after each global interation
                self.dt = []
                self.total_dt = []
                for param in self.model.parameters():
                    self.dt.append(torch.zeros_like(param.data))
                    self.total_dt.append(torch.zeros_like(param.data))

                # Aggregate grads of client.
                for edge in self.edges:
                    edge.get_full_grad()
                self.aggregate_grads()

                # Richardson
                for r in range(self.local_epochs):
                    self.send_dt()
                    self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                    for edge in self.selected_edges:
                        edge.gethessianproduct(self.local_epochs, glob_iter)
                    self.aggregate_dt()

                self.aggregate_newton()

        elif self.algorithm == "Newton2":
            for glob_iter in range(self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                self.send_parameters()
                self.evaluate()

                for edge in self.edges:
                    edge.get_full_grad()
                self.aggregate_grads()

                hess = self.aggregate_hessians()
                inverse_hess = torch.inverse(hess)
                grads = []
                for param in self.model.parameters():
                    grads.append(param.grad.clone().detach())
                grads_as_vector = torch.cat([-grads[0].data, -grads[1].data.view(1, 1)], 1)
                direction = torch.matmul(grads_as_vector, inverse_hess)
                weights_direction = direction[0, 0:-1].view(grads[0].shape)
                bias_direction = direction[0, -1].view(grads[1].shape)

                for param, d in zip(self.model.parameters(), [weights_direction, bias_direction]):
                    param.data.add_(self.eta * d)

                # self.selected_edges = self.select_edges(glob_iter, self.num_edges)
                # for edge in self.selected_edges:
                #     edge.get_hessian(self.local_epochs, glob_iter)

                # self.aggregate_parameters()

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

    def aggregate_hessians(self):
        aggregated_hessians = None
        i = 0
        total_samples = 0
        for i, edge in enumerate(self.edges):
            hess = edge.send_hessian()
            total_samples += edge.train_samples
            if aggregated_hessians is None:
                aggregated_hessians = hess
            else:
                aggregated_hessians.add_(hess)
        return aggregated_hessians / (i + 1 + 1e-6)
