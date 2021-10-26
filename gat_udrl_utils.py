# for GAT_UDRL
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import DGLDataset
from dgl.nn import GATConv

import pandas as pd

# one layer GAT
class GAT(nn.Module):
    def __init__(self, in_feats, out_feats_gat, num_heads, h_feats, num_classes):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, out_feats_gat, num_heads)
        # encode reward-to-go
        self.fc1 = nn.Linear(1, h_feats)
        # predict action
        self.fc2 = nn.Linear((out_feats_gat*num_heads+h_feats), num_classes)
    
    def forward(self, g, in_feat, reward_to_go, cur_node):
        # graph goes through GAT
        h = self.gat1(g, in_feat)
        h = torch.sigmoid(h)
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
Data Loader.
"""
class GAT_Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='GAT_dataset')

    def process(self):
        edges = pd.read_csv('./random/graph_structure.csv')
        properties = pd.read_pickle('./random/sequential_samples.pkl')
        self.graphs = []
        self.labels = []
        self.returns = []
        self.cur_items = []

        num_nodes = edges['num_nodes'][0]
        src = eval(edges['src_nodes'][0])
        dst = eval(edges['dst_nodes'][0])
        for i in range(len(properties)):
            cur_row = properties.loc[i]
            feats_multi, cur_nodes_multi = cur_row['feats'], cur_row['cur_node_one_hot']
            assert len(feats_multi) == len(cur_nodes_multi)
            labels_multi, returns_multi = cur_row['labels'], cur_row['returns']
            assert len(labels_multi) == len(returns_multi)
            assert len(feats_multi) == len(labels_multi)
            for j in range(len(labels_multi)):
                g = dgl.graph((src, dst), num_nodes=num_nodes)
                g = dgl.add_self_loop(g)
                g.ndata['feat'] = torch.from_numpy(feats_multi[j])
                self.graphs.append(g)
                self.labels.append(labels_multi[j])
                self.returns.append(returns_multi[j])
                self.cur_items.append(cur_nodes_multi[j])

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        self.returns = torch.LongTensor(self.returns)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.returns[i], self.cur_items[i]

    def __len__(self):
        return len(self.graphs)