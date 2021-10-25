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
    def __init__(self, in_feats, out_feats_gat, num_heads, h_feats, out_feats):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, out_feats_gat, num_heads, activation=nn.Sigmoid)
        # encode reward-to-go
        self.fc1 = nn.Linear(1, h_feats)
        # predict action
        self.fc2 = nn.Linear((out_feats_gat*num_heads+h_feats), out_feats)
    
    def forward(self, g, in_feat, reward_to_go, cur_node):
        # graph goes through GAT
        h = self.gat1(g, in_feat)
        # concatenate multiple heads
        h = h.view(g.num_nodes(),-1)
        # select the particular node using the one-hot vector
        h = torch.sum(torch.multiply(h, cur_node), dim=1)
        # encode reward_to_go
        h_r = self.fc1(reward_to_go)
        h_r = F.relu(h_r)
        h = torch.cat((h, h_r))
        # prediction layer
        h = F.softmax(self.fc2(h))
        return h


"""
Data Loader.
"""
class GAT_Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='GAT_dataset')

    def process(self):
        edges = pd.read_csv('./random/graph_structure.csv')
        properties = pd.read_csv('./random/sequential_samples.csv')
        self.graphs = []
        self.labels = []
        self.returns = []
        self.cur_items = []

        num_nodes = edges['num_nodes'][0]
        src = eval(edges['src_nodes'][0])
        dst = eval(edges['dst_nodes'][0])
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.ndata['feat'] = 
        self.graphs.append(g)
        self.labels.append(label)

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)