# generate training dataset from 'node_association' environment.

import numpy as np
import gym
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import pandas as pd
import csv

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx
import time
from env_utils import *

data_config = edict({'N_Episodes': 100,
                    })

env_config = edict({'N_RUs':20,
                 'N_DUs':4,
                 })

env = gym.make("gym_graph_node_association:graph_association-v0")
env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
n_actions = env.action_space.n

"""
Dataset 1: Random actions
"""

# i. number of nodes; src nodes and dst nodes - generate a complete graph
n_nodes = env_config.N_RUs
# complete graph
c_graph = nx.complete_graph(n_nodes)
c_edges = list(c_graph.edges)
src = [item[0] for item in c_edges]
dst = [item[1] for item in c_edges]
# save graph_structure
graph_structure = {'num_nodes':n_nodes,
                    'src_nodes': src,
                    'dst_nodes': dst,
                  }
with open('./random/graph_structure.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, graph_structure.keys())
    w.writeheader()
    w.writerow(graph_structure)

# ii. node features, actions, reward-to-go
data_pd = pd.DataFrame()
dataset_random = {'feats': [None],
                    'cur_node_one_hot': [None],
                    'labels': [None],
                    'returns': [None],}

t_0 = time.time()
for idx in range(data_config.N_Episodes):
    done = False
    env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
    # ndata in graph
    feats = get_node_feat(env)
    nodes_actions = np.zeros((n_nodes, env.action_space.n))
    feats_with_actions = []
    actions = []
    rewards = []
    cur_nodes = []
    while not done:
        feats_with_actions.append(np.concatenate((feats, nodes_actions), axis=1))
        # a one-hot vector representing the current node, corresponding with labels
        current_node = np.zeros(env.n_nodes[0])
        current_node[env.cur_item] = 1
        a = env.action_space.sample()
        # encode the action taken
        nodes_actions[env.cur_item, a] = 1 
        _, r, done, _ = env.step(a)       
        actions.append(a)
        rewards.append(r)
        cur_nodes.append(list(current_node))
    returns = reward_to_go(rewards)
    dataset_random['feats'] = feats_with_actions
    dataset_random['cur_node_one_hot'] = cur_nodes
    dataset_random['labels'] = actions
    dataset_random['returns'] = returns
    data_pd = data_pd.append(dataset_random, ignore_index=True)

# save dataset
data_pd.to_pickle('./random/sequential_samples.pkl')
print('Time to generate %f samples is %f' % (data_config.N_Episodes, time.time()-t_0))