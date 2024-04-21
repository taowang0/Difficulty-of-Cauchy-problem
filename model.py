import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import Parameter


class MLP(nn.Module):
    def __init__(self, dim, activation):
        super(MLP, self).__init__()
        self.dim = dim
        self.linear_list = nn.ModuleList()
        for i in range(len(dim) - 2):
            self.linear_list.append(nn.Linear(dim[i+1], dim[i+2]))

        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation must be one of relu, tanh, and sigmoid.')

    def forward(self, x):
        out = x
        for i in range(len(self.dim) - 3):
            out = self.linear_list[i](out)
            out = self.activation(out)
        out = self.linear_list[len(self.dim) - 3](out)
        return out
