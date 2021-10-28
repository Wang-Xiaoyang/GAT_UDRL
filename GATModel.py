import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import DGLDataset
from dgl.nn import GATConv

import pandas as pd
import numpy as np


"""
GAT model for GAT_UDRL_1 model. The difference is in how to encode the reward-to-go.
"""
# multi-layer GAT
class GAT1(nn.Module):
    def __init__(self, in_feats, out_feats_gat, num_heads, h_feats, num_classes):
        super(GAT1, self).__init__()
        self.gat1 = GATConv(in_feats, out_feats_gat, num_heads)
        self.gat2 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        self.gat3 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        # self.gat4 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        # encode reward-to-go
        self.fc1 = nn.Linear(1, h_feats)
        # predict action
        self.fc2 = nn.Linear((out_feats_gat*num_heads+h_feats), num_classes)
    
    def forward(self, g, in_feat, reward_to_go, cur_node):
        # graph goes through GAT
        h = self.gat1(g, in_feat)
        h = torch.sigmoid(h)
        h = h.view(g.num_nodes(),-1)
        h = self.gat2(g, h)
        h = torch.sigmoid(h)
        h = h.view(g.num_nodes(),-1)
        h = self.gat3(g, h)
        h = torch.sigmoid(h)
        # h = h.view(g.num_nodes(),-1)
        # h = self.gat4(g, h)
        # h = torch.sigmoid(h)
        # concatenate multiple heads
        h = h.view(g.num_nodes(),-1)
        h = h.reshape(cur_node.size()[0], cur_node.size()[1], h.size()[-1])
        # select the particular node using the one-hot vector
        h = torch.matmul(cur_node.unsqueeze(dim=1), h)
        h = h.squeeze(dim=1)
        # encode reward_to_go
        h_r = self.fc1(reward_to_go.unsqueeze(dim=1))
        h_r = F.relu(h_r)
        h = torch.cat((h, h_r), dim=1)
        # prediction layer
        h = self.fc2(h)
        return h

"""
GAT model for GAT_UDRL_2 model. The difference is in how to encode the reward-to-go.
In this mode, reward-to-go is part of the node features of graphs, rather than having
seperate head and seperate encoder.
"""
class GAT2(nn.Module):
    def __init__(self, in_feats, out_feats_gat, num_heads, h_feats, num_classes):
        super(GAT2, self).__init__()
        self.gat1 = GATConv(in_feats, out_feats_gat, num_heads)
        self.gat2 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        self.gat3 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        # self.gat4 = GATConv(out_feats_gat*num_heads, out_feats_gat, num_heads)
        # encode reward-to-go
        self.fc1 = nn.Linear(1, h_feats)
        # predict action
        # self.fc2 = nn.Linear((out_feats_gat*num_heads+h_feats), num_classes)
        self.fc2 = nn.Linear((out_feats_gat*num_heads), num_classes)
    
    def forward(self, g, in_feat, reward_to_go, cur_node):
        # graph goes through GAT
        h = self.gat1(g, in_feat)
        h = torch.sigmoid(h)
        h = h.view(g.num_nodes(),-1)
        h = self.gat2(g, h)
        h = torch.sigmoid(h)
        h = h.view(g.num_nodes(),-1)
        h = self.gat3(g, h)
        h = torch.sigmoid(h)
        # h = h.view(g.num_nodes(),-1)
        # h = self.gat4(g, h)
        # h = torch.sigmoid(h)
        # concatenate multiple heads
        h = h.view(g.num_nodes(),-1)
        h = h.reshape(cur_node.size()[0], cur_node.size()[1], h.size()[-1])
        # select the particular node using the one-hot vector
        h = torch.matmul(cur_node.unsqueeze(dim=1), h)
        h = h.squeeze(dim=1)
        """
        # encode reward_to_go
        h_r = self.fc1(reward_to_go.unsqueeze(dim=1))
        h_r = F.relu(h_r)
        h = torch.cat((h, h_r), dim=1)
        """
        # prediction layer
        h = self.fc2(h)
        return h