# for GAT_UDRL
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import DGLDataset
from dgl.nn import GATConv

import pandas as pd
import numpy as np

# graph shortest path env

dir_name = 'shortest_path/random_walk'

"""
Data loader for GAT_UDRL_1 model.
"""
class GAT_Dataset_1(DGLDataset):
    def __init__(self):
        super().__init__(name='GAT_dataset')

    def process(self):
        self.dir_name = dir_name
        edges = pd.read_pickle('./' + self.dir_name + '/graph_structure.pkl')
        properties = pd.read_pickle('./' + self.dir_name + '/sequential_samples.pkl')
        assert len(edges) == len(properties)
        self.graphs = []
        self.labels = []
        self.returns = []
        self.cur_items = []

        for i in range(len(properties)):
            num_nodes = edges['num_nodes'][i]
            src = edges['src_nodes'][i]
            dst = edges['dst_nodes'][i]
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
        self.returns = torch.FloatTensor(self.returns)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.returns[i], self.cur_items[i]

    def __len__(self):
        return len(self.graphs)

"""
Data loader for GAT_UDRL_2 model.
"""
class GAT_Dataset_2(DGLDataset):
    def __init__(self):
        super().__init__(name='GAT_dataset')

    def process(self):
        self.dir_name = dir_name
        edges = pd.read_csv('./' + self.dir_name + '/graph_structure.csv')
        properties = pd.read_pickle('./' + self.dir_name + '/sequential_samples.pkl')
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
                returns_so_far = np.zeros(num_nodes)
                returns_so_far[0:j+1] = returns_multi[0:j+1].copy()
                returns_so_far = returns_so_far / 10 # scaling
                feats_with_returns = torch.from_numpy(np.concatenate((feats_multi[j], np.expand_dims(returns_so_far, axis=1)), axis=1))
                g.ndata['feat_with_returns'] = feats_with_returns
                self.graphs.append(g)
                self.labels.append(labels_multi[j])
                self.returns.append(returns_multi[j])
                self.cur_items.append(cur_nodes_multi[j])

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        self.returns = torch.FloatTensor(self.returns)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.returns[i], self.cur_items[i]

    def __len__(self):
        return len(self.graphs)