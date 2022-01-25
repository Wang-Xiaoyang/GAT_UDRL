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

dataset_type = 'distance-first' # random, distance-first or consumption-first

data_config = edict({'N_Episodes': 100,
                    })

env_config = edict({'N_RUs':20,
                 'N_DUs':2,
                 })

env = gym.make("gym_graph_node_association:graph_association-v0")
env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
n_actions = env.action_space.n

# """
# Dataset 1: Random actions
# """
if dataset_type == 'random':
    # state (type 0 node): [node feature (RU requirement), x, y (location)]
    # for each state, predict its action (which type 1 node to connect) - classification problem
    n_nodes = env_config.N_RUs

    # node features, actions (labels), reward-to-go
    data_pd = pd.DataFrame()
    dataset_random = {'feats': [None],
                        'cur_node_one_hot': [None],
                        'labels': [None],
                        'returns': [None],}

    t_0 = time.time()
    for idx in range(data_config.N_Episodes):
        done = False
        env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
        # states
        feats = get_state_from_env(env)
        actions = []
        rewards = []
        cur_nodes = []
        while not done:
            # feats_with_actions.append(np.concatenate((feats, nodes_actions), axis=1))
            # a one-hot vector representing the current node, corresponding with labels
            current_node = np.zeros(env.n_nodes[0])
            current_node[env.cur_item] = 1

            a = env.action_space.sample()            
            _, r, done, _ = env.step(a)     
            actions.append(a)
            rewards.append(r)
            cur_nodes.append(list(current_node))
        returns = reward_to_go(rewards)
        dataset_random['feats'] = feats
        dataset_random['cur_node_one_hot'] = cur_nodes
        dataset_random['labels'] = actions
        dataset_random['returns'] = returns
        data_pd = data_pd.append(dataset_random, ignore_index=True)

    # save dataset
    data_pd.to_pickle('./random/sequential_samples_bert.pkl')
    print('Time to generate %d samples is %f' % (data_config.N_Episodes, time.time()-t_0))


# """
# Dataset 2: Choose first the type 1 node with the shortest distance.
# distance-first
# """
if dataset_type == 'distance-first':
    # state (type 0 node): [node feature (RU requirement), x, y (location)]
    # for each state, predict its action (which type 1 node to connect) - classification problem
    n_nodes = env_config.N_RUs

    # node features, actions (labels), reward-to-go
    data_pd = pd.DataFrame()
    dataset_random = {'feats': [None],
                        'cur_node_one_hot': [None],
                        'labels': [None],
                        'returns': [None],}

    t_0 = time.time()
    for idx in range(data_config.N_Episodes):
        done = False
        env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
        # states
        feats = get_state_from_env(env)
        actions = []
        rewards = []
        cur_nodes = []
        while not done:
            # a one-hot vector representing the current node, corresponding with labels
            current_node = np.zeros(env.n_nodes[0])
            current_node[env.cur_item] = 1
            dists = np.zeros(2)
            for a_i in range(n_actions):
                dist = env.calculate_distance(a_i)
                dists[a_i] = dist
            a_h = dists.argmin()
            # check if this action is valid
            if env.if_action_valid(a_h) == True:
                a = a_h
            else:
                # to improve here
                a = env.action_space.sample() 

            _, r, done, _ = env.step(a)     
            actions.append(a)
            rewards.append(r)
            cur_nodes.append(list(current_node))
        returns = reward_to_go(rewards)
        dataset_random['feats'] = feats
        dataset_random['cur_node_one_hot'] = cur_nodes
        dataset_random['labels'] = actions
        dataset_random['returns'] = returns
        data_pd = data_pd.append(dataset_random, ignore_index=True)

    # save dataset
    data_pd.to_pickle('./distance-first/sequential_samples_bert.pkl')
    print('Time to generate %d samples is %f' % (data_config.N_Episodes, time.time()-t_0))


"""
Dataset 3: Choose first the type 1 node with the lowest power consumption.
consumption-first
"""
if dataset_type == 'consumption-first':
    # state (type 0 node): [node feature (RU requirement), x, y (location)]
    # for each state, predict its action (which type 1 node to connect) - classification problem
    n_nodes = env_config.N_RUs

    # node features, actions (labels), reward-to-go
    data_pd = pd.DataFrame()
    dataset_random = {'feats': [None],
                        'cur_node_one_hot': [None],
                        'labels': [None],
                        'returns': [None],}

    t_0 = time.time()
    for idx in range(data_config.N_Episodes):
        done = False
        env.reset(n_nodes=[env_config.N_RUs, env_config.N_DUs])
        # states
        feats = get_state_from_env(env)
        actions = []
        rewards = []
        cur_nodes = []
        while not done:
            # feats_with_actions.append(np.concatenate((feats, nodes_actions), axis=1))
            # a one-hot vector representing the current node, corresponding with labels
            current_node = np.zeros(env.n_nodes[0])
            current_node[env.cur_item] = 1
            consumptions = np.zeros(2)
            for a_i in range(n_actions):
                adj_mat = env.adjacency_mat.copy()
                consumption_before = env.capacity_consumption(adj_mat)
                adj_mat[env.cur_item, a_i] = 1
                consumption_after = env.capacity_consumption(adj_mat)
                consumption = consumption_after - consumption_before
                consumptions[a_i] = consumption
            a_h = consumptions.argmin()
            # check if this action is valid
            if env.if_action_valid(a_h) == True:
                a = a_h
            else:
                # to improve here
                a = env.action_space.sample()

            _, r, done, _ = env.step(a)     
            actions.append(a)
            rewards.append(r)
            cur_nodes.append(list(current_node))
        returns = reward_to_go(rewards)
        dataset_random['feats'] = feats
        dataset_random['cur_node_one_hot'] = cur_nodes
        dataset_random['labels'] = actions
        dataset_random['returns'] = returns
        data_pd = data_pd.append(dataset_random, ignore_index=True)

    # save dataset
    data_pd.to_pickle('./consumption-first/sequential_samples_bert.pkl')
    print('Time to generate %d samples is %f' % (data_config.N_Episodes, time.time()-t_0))