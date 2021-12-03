# generate training dataset for the 'shortest path' problem in graphs.

from networkx.classes.function import is_empty
import numpy as np
import gym
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import pandas as pd
import csv
import random
from itertools import groupby

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import networkx as nx
import time
from env_utils import *
from shortest_path_utils import *

data_config = edict({'N_Graphs': 1000,
                    })

# N_steps is the max_steps for random walk.
env_config = edict({'N_Nodes': 20,
                 'N_Steps': 10,
                 'Edge_Sparsity': 0.1
                 })

env = gym.make('gym_graph_shortest_path:graph_shortest_path-v0')
n_actions = env.action_space.n
env.reset(n_nodes = env_config.N_Nodes, 
            edge_sparsity = env_config.Edge_Sparsity, max_steps = env_config.N_Steps, random_seed = None)

"""
Dataset 1: Random Walk on undirectional graph
Not using one-hot representations
"""
# # i. number of nodes; src nodes and dst nodes - generate a complete graph
# n_nodes = env_config.N_Nodes

# # ii. node features, actions, reward-to-go
# data_pd = pd.DataFrame()
# dataset_random = {'feats': [None],
#                     'cur_node_one_hot': [None],
#                     'labels': [None],
#                     'returns': [None],}

# graph_structure_pd = pd.DataFrame()
# graph_structure = {'num_nodes': None,
#                     'src_nodes': None,
#                     'dst_nodes': None,
#                   }

# reach_goal = 0

# t_0 = time.time()
# for idx in range(data_config.N_Graphs):
#     done = False
#     env.reset(n_nodes = env_config.N_Nodes, 
#             edge_sparsity = env_config.Edge_Sparsity, max_steps = env_config.N_Steps, random_seed = None)

#     graph_structure = {'num_nodes':n_nodes,
#                     'src_nodes': env.src,
#                     'dst_nodes': env.dst,
#                   }
#     graph_structure_pd = graph_structure_pd.append(graph_structure, ignore_index=True)

#     # ndata in graph; features (node id): one-hot encoding
#     feats = np.array(list(range(n_nodes))) / n_nodes # normalized to 1
#     feats = np.expand_dims(feats, axis=1)
#     nodes_actions = -1 * np.ones((n_nodes, 1))
#     node_goal = np.zeros((n_nodes, 1))
#     node_goal[env.goal_idx] = 1
#     node_goal[env.start_idx] = -1
#     feats_with_actions = []
#     actions = []
#     rewards = []
#     cur_nodes = []
#     while not done:
#         feats_with_actions.append(np.concatenate((feats, node_goal, nodes_actions), axis=1))
#         # a vector representing the current node, corresponding with labels
#         current_node = np.zeros(env.n_nodes)
#         current_node[env.ob] = 1
#         #random walk
#         a_list = env.get_neighbor_nodes(env.ob)
#         # if there are other choices, avoid staying at current node
#         if len(a_list) > 1:
#             a_list.remove(env.ob)
#         a = random.choice(a_list)
#         # if agent stays in a node, terminate
#         if a == env.ob:
#             done = True
#             actions.append(a)
#             # a = a / env.action_space.n # normalisation
#             # nodes_actions[env.ob] = a
#             r = env.r_min - sum(rewards) # so that the reward to go is [-10, -9,...]
#         else:
#             # encode the action taken; add this action to node state
#             nodes_actions[env.ob] = a / env.action_space.n
#             _, r, done, _ = env.step(a)
#             actions.append(a)
#             if r == 0:
#                 reach_goal += 1  
#         rewards.append(r)
#         cur_nodes.append(list(current_node))
#     returns = reward_to_go(rewards)
#     dataset_random['feats'] = feats_with_actions
#     dataset_random['cur_node_one_hot'] = cur_nodes
#     dataset_random['labels'] = actions
#     dataset_random['returns'] = returns
#     data_pd = data_pd.append(dataset_random, ignore_index=True)

# # save dataset
# data_pd.to_pickle('./shortest_path/random_walk/sequential_samples.pkl')
# graph_structure_pd.to_pickle('./shortest_path/random_walk/graph_structure.pkl')
# print('Time to generate %d samples is %f' % (data_config.N_Graphs, time.time()-t_0))
# print('Random walk reaches the goal in %d / %d graphs within %d steps in the env' % (reach_goal, data_config.N_Graphs, env_config.N_Steps))



"""
Dataset 2: Breadth First Search (BFS) on undirectional graph
Not using one-hot representations
"""
# i. number of nodes; src nodes and dst nodes - generate a complete graph
n_nodes = env_config.N_Nodes

# ii. node features, actions, reward-to-go
data_pd = pd.DataFrame()
dataset_random = {'feats': [None],
                    'cur_node_one_hot': [None],
                    'labels': [None],
                    'returns': [None],}

graph_structure_pd = pd.DataFrame()
graph_structure = {'num_nodes': None,
                    'src_nodes': None,
                    'dst_nodes': None,
                  }

reach_goal = 0
# values = 0
# len_bfs = 0

t_0 = time.time()
for idx in range(data_config.N_Graphs):
    done = False
    env.reset(n_nodes = env_config.N_Nodes, 
            edge_sparsity = env_config.Edge_Sparsity, max_steps = env_config.N_Steps, random_seed = None)

    graph_structure = {'num_nodes':n_nodes,
                    'src_nodes': env.src,
                    'dst_nodes': env.dst,
                  }
    graph_structure_pd = graph_structure_pd.append(graph_structure, ignore_index=True)

    # ndata in graph; features (node id): one-hot encoding
    feats = np.array(list(range(n_nodes))) / n_nodes # normalized to 1
    feats = np.expand_dims(feats, axis=1)
    nodes_actions = -1 * np.ones((n_nodes, 1))
    node_goal = np.zeros((n_nodes, 1))
    node_goal[env.goal_idx] = 1
    node_goal[env.start_idx] = -1
    feats_with_actions = []
    actions = []
    rewards = []
    cur_nodes = []
    # get actions
    result_flag, actions_bfs = bfs_shortest_path(env.start_idx, env.goal_idx, env)
    actions_bfs = actions_bfs[1:]
    if result_flag == 0:
        # start node = goal node, skip
        continue
    else:
        while not done:
            feats_with_actions.append(np.concatenate((feats, node_goal, nodes_actions), axis=1))
            # a vector representing the current node, corresponding with labels
            current_node = np.zeros(env.n_nodes)
            current_node[env.ob] = 1
            # fetch actions
            if len(actions_bfs) == 0:
                a = env.ob
            else:
                a = actions_bfs[0]
                actions_bfs.pop(0)
            
            nodes_actions[env.ob] = a / env.action_space.n
            # encode the action taken
            _, r, done, _ = env.step(a)
            actions.append(a)            
            if r == 1:
                reach_goal += 1  
            rewards.append(r)
            cur_nodes.append(list(current_node))
    returns = reward_to_go(rewards)
    dataset_random['feats'] = feats_with_actions
    dataset_random['cur_node_one_hot'] = cur_nodes
    dataset_random['labels'] = actions
    dataset_random['returns'] = returns
    data_pd = data_pd.append(dataset_random, ignore_index=True)

# save dataset
data_pd.to_pickle('./shortest_path/bfs/sequential_samples.pkl')
graph_structure_pd.to_pickle('./shortest_path/bfs/graph_structure.pkl')
print('Time to generate %d samples is %f' % (data_config.N_Graphs, time.time()-t_0))
print('BFS reaches the goal in %d / %d graphs within %d steps in the env' % (reach_goal, data_config.N_Graphs, env_config.N_Steps))