import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, nlin1, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.fc1 = nn.Linear(nout, nlin1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class phi_pro(nn.Module):
    def __init__(self, n_nodes, n_out, n_feat, n_class):
        super().__init__()
        self.fc1 = nn.Linear(n_nodes, n_out)
        self.fc2 = nn.Linear(n_feat, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.sum(axis=1, keepdims=True)
        x = F.softmax(self.fc2(x))
        return x
