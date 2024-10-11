import sys
from functools import reduce
from math import factorial
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx



class SoftmaxMarkovChain(nn.Module):
    NAME = 'SoftmaxMarkovChain'
    def __init__(self, N, M):
        super(SoftmaxMarkovChain, self).__init__()
        self.N = N
        self.M = M
        self.W = nn.Parameter(torch.randn(self.N, self.N))
        # print(f'Initial P in the neural network:\n{torch.nn.functional.softmax(self.W, dim=0, dtype=torch.double)}')
        self.double()

    def self_loop_regulaizer_loss(self):
        P = torch.nn.functional.softmax(self.W, dim=0, dtype=torch.double)
        return torch.sum(torch.diag(P))

    def forward(self, walker_states_proba_vector):
        # walker_states_proba_vector is in format (M, N)
        # Perform the multiplication in transpose form (as in torch format) (M, N) x (N, N) = (M, N)
        P = torch.nn.functional.softmax(self.W, dim=0, dtype=torch.double)
        return torch.matmul(walker_states_proba_vector, P.T)
    
    def get_internal_model(self):
        return torch.nn.functional.softmax(self.W.detach().clone(), dim=0, dtype=torch.double)


class SteadyStateSoftmaxMarkovChain(SoftmaxMarkovChain):
    NAME = 'SteadyStateSoftmaxMarkovChain'
    def __init__(self, N, M):
        super(SteadyStateSoftmaxMarkovChain, self).__init__(N, M)
        self.W = nn.Parameter(torch.randn(self.N))
        self.double()

    def forward(self):
        P = torch.nn.functional.softmax(self.W, dim=0, dtype=torch.double)
        return P.repeat(self.M, 1).T  # Return in  (N, M) format

    def self_loop_regulaizer_loss(self):
        p = torch.nn.functional.softmax(self.W, dim=0, dtype=torch.double)

        # Move away from edge of polytope 
        edge_loss = 1 / torch.prod(p * (1-p)) * 0.1
        entropy_loss = -torch.sum(p * torch.log(p)) *0.1

        return edge_loss + entropy_loss
    
    def get_internal_model(self):
        return self
    

class MultiplePolicySteadyStateSoftmaxMarkovChain(SteadyStateSoftmaxMarkovChain):
    NAME = 'MultiplePolicySteadyStateSoftmaxMarkovChain'
    def __init__(self, N, M):
        super(MultiplePolicySteadyStateSoftmaxMarkovChain, self).__init__(N, M)
        self.W = nn.Parameter(torch.randn(self.N, self.M))
        self.double()

    def forward(self):
        # walker_states_proba_vector is in format (M, N)
        P = torch.nn.functional.softmax(self.W, dim = 0, dtype=torch.double) 
        return P  # Return in  (N, M) format

class MLPWalkerPolicy(nn.Module):
    """
    General walker policy as a neural network
    """
    NAME = 'MLPWalkerPolicy'
    def __init__(self, N, M, layers=[64, 64]) -> None:
        super(MLPWalkerPolicy, self).__init__()
        self.N = N
        self.M = M
        layers = [self.N] + layers + [self.N]
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.double()
        
    def forward(self, walker_states_proba_vector):
        for layer in self.layers[:-1]:
            walker_states_proba_vector = layer(walker_states_proba_vector)
            walker_states_proba_vector = torch.relu(walker_states_proba_vector)
        walker_states_proba_vector = torch.nn.functional.softmax(self.layers[-1](walker_states_proba_vector), dim=1)
        # print(f'walker_states_proba_vector:\n{walker_states_proba_vector}')
        return walker_states_proba_vector
    
    def get_internal_model(self):
        return self