# from networkx.classes.function import nodes
import numpy as np
import gym
import torch
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import gc

# import dgl
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn.pytorch import GraphConv

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
    # add feature: location of type 1 nodes
    nodes_x_1, nodes_y_1 = env.nodes_x[n_nodes:], env.nodes_y[n_nodes:]
    nodes_x_1 = np.repeat(nodes_x_1.transpose(), n_nodes, axis=0)
    nodes_y_1 = np.repeat(nodes_y_1.transpose(), n_nodes, axis=0)
    nodes_fea = np.concatenate((nodes_fea, nodes_x_1, nodes_y_1), axis=1)
    nodes_fea = nodes_fea / 10
    return nodes_fea

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

# BERT UDRL
# node features: [node feature, x, y (location), x, y (location for type #1 nnodes)]
# This state is not Markovian! Meaning, given more information, it would help (say, the history of actions, or type 1 nodes remaining capacities)
# Add information: for each step, add type 1 node remaining capacity
# Assuming that is some environment information shared with type 0 nodes
def get_state_from_env(env, nodes_fea):
    if len(nodes_fea) == 0: # initialize
        n_nodes = env.n_nodes[0]        
        # value
        nodes_type_0_v = np.transpose(env.fea.copy())
        nodes_v = np.expand_dims(nodes_type_0_v, axis=1)
        # location
        nodes_x, nodes_y = env.nodes_x[0:n_nodes], env.nodes_y[0:n_nodes]
        nodes_fea = np.concatenate((nodes_v, nodes_x, nodes_y), axis=1)
        # add feature: location of type 1 nodes
        nodes_x_1, nodes_y_1 = env.nodes_x[n_nodes:], env.nodes_y[n_nodes:]
        nodes_x_1 = np.repeat(nodes_x_1.transpose(), n_nodes, axis=0)
        nodes_y_1 = np.repeat(nodes_y_1.transpose(), n_nodes, axis=0)
        nodes_fea = np.concatenate((nodes_fea, nodes_x_1, nodes_y_1), axis=1)
        # type 1 node remaining capacities
        remaining_capacity = np.zeros([n_nodes, env.n_nodes[1]])
        nodes_fea = np.concatenate((nodes_fea, remaining_capacity), axis=1)       
    else: # during an episode
        nodes_fea[env.cur_item,-env.n_nodes[1]:] = env.remaining_capacity / env.C * 10 # for later normalizetion = /10
    return nodes_fea