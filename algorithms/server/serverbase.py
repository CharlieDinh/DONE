import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class ServerBase:
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate , L, num_glob_iters, local_epochs, optimizer, num_edges, times):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.edges = []
        self.selected_edges = []
        self.num_edges = num_edges
        self.L = L
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc= [], [], []
        self.times = times
        
    def aggregate_grads(self):
        assert (self.edges is not None and len(self.edges) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for edge in self.edges:
            self.add_grad(edge, edge.train_samples / self.total_train_samples)
    
    def send_parameters(self):
        assert (self.edges is not None and len(self.edges) > 0)
        for edge in self.edges:
            edge.set_parameters(self.model)

    def add_parameters(self, edge, ratio):
        model = self.model.parameters()
        for server_param, edge_param in zip(self.model.parameters(), edge.get_parameters()):
            server_param.data = server_param.data + edge_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.edges is not None and len(self.edges) > 0)
        total_train = 0
        for edge in self.selected_edges:
            total_train += edge.train_samples

        for edge in self.selected_edges:
            self.add_parameters(edge, edge.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_edges(self, round, num_edges):
        if(num_edges == len(self.edges)):
            print("All edges are selected")
            return self.edges

        num_edges = min(num_edges, len(self.edges))
        np.random.seed(round)
        return np.random.choice(self.edges, num_edges, replace=False)

            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.L) + "_" + str(self.num_edges) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.edges:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.edges]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.edges:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.edges]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)
