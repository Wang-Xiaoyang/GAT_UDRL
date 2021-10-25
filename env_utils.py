import numpy as np
import gym
import torch
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import gc

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# GAT_UDRL: fully connect graph with all type 0 nodes
# env state to node featurs in graph
# node feature (graph.ndata['fea']): [value; location (x, y)]
def get_node_feat(env):
    n_nodes = env.n_nodes[0]
    # value
    nodes_type_0_v = np.transpose(env.fea.copy())
    nodes_v = np.expand_dims(nodes_type_0_v, axis=1)
    # location
    nodes_x, nodes_y = env.nodes_x[0:n_nodes], env.nodes_y[0:n_nodes]
    nodes_fea = np.concatenate((nodes_v, nodes_x, nodes_y), axis=1)
    return nodes_fea / 10

def reward_to_go(rewards, gamma=1.0):
    """Reward to go (discounted and undiscounted)
    Args:
    Returns: a sequence of rewards in a trajectory
    """
    # get returns of states from a trajectory
    # R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    returns = rewards.copy()
    for i in reversed(range(len(returns)-1)):
        r = returns[i] + gamma * returns[i+1]
        returns[i] = r
    return returns 



# env state to graph features
def state_to_node_fea(env):
    # get node features from env states; normalize
    # 1. one-hot representation for node types
    data = np.asarray([['type 0'], ['type 1']])
    encoder = OneHotEncoder(sparse=False)
    onehot = encoder.fit_transform(data)
    nodes_type_0_t = onehot[0] # type
    nodes_type_0_t = np.reshape(np.tile(nodes_type_0_t, env.n_nodes[0]), (env.n_nodes[0], -1)) # n_nodes_type_0 * feature_len
    nodes_type_1_t = onehot[1]
    nodes_type_1_t = np.reshape(np.tile(nodes_type_1_t, env.n_nodes[1]), (env.n_nodes[1], -1))
    nodes_t = np.concatenate((nodes_type_0_t, nodes_type_1_t))

    # 2. type 0 values; type 1: remaining capacity
    nodes_type_0_v = np.transpose(env.fea.copy())
    nodes_type_1_v = np.transpose(env.remaining_capacity)
    nodes_v = np.concatenate((nodes_type_0_v, nodes_type_1_v))
    nodes_v = np.expand_dims(nodes_v, axis=1)
    nodes_v = nodes_v / env.C

    # concatenate
    nodes_fea = np.concatenate((nodes_t, nodes_v), axis=1)

    # 3. current node (in terms of MDP action) is 1, others are 0
    current_node = np.zeros((env.n_nodes_total, 1))
    current_node[env.cur_item] = 1
    nodes_fea = np.concatenate((nodes_fea, current_node), axis=1)

    # 4. location
    nodes_x, nodes_y = env.nodes_x / 10, env.nodes_y / 10
    nodes_fea = np.concatenate((nodes_fea, nodes_x, nodes_y), axis=1)

    return nodes_fea

def adjacency_vbpp_to_graph(adjacency_mat):
    # this function transfers the adjacency matrix of the node association env, 
    # which has the size [n_type_0_nodes, n_type_1_nodes], to the format of pair of nodes [u, v]
    # for defining a graph
    # u: source nodes; v: destination nodes
    n_type_0_nodes = adjacency_mat.shape[0]
    a = np.sum(adjacency_mat, axis=1)
    u = np.where(a == 1)[0]
    v = np.zeros(len(u), dtype=int)
    idx = 0
    for i in u:
        v[idx] = np.where(adjacency_mat[i, :] == 1)[0][0] + n_type_0_nodes
    return u, v