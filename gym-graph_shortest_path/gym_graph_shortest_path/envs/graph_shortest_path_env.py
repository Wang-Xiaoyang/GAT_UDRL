from dgl import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import networkx as nx
import dgl
import torch

import math
import numpy as np
from numpy.core.fromnumeric import size

class GraphShortestPathEnv(gym.Env):
    """
    An environment for node association problems in graph
    considering multiple constraints.
    """

    def __init__(self):
        super(GraphShortestPathEnv, self).__init__()
        self.n_nodes = 20
        self.edge_sparsity = 0.1
        self.max_steps = 10
        # start idx and goal idx
        self.start_idx = 0
        self.goal_idx = self.n_nodes - 1
        
        # p(s0) - featuers of type 0 nodes - in each episode, randomly sample form a Gaussian distribution
        # simulation of user mobility in this area
        np.random.seed()
        c_graph = nx.complete_graph(self.n_nodes)
        c_edges = list(c_graph.edges)
        src_complete = [item[0] for item in c_edges]
        dst_complete = [item[1] for item in c_edges]
        n_edges = int(np.ceil(len(src_complete) * self.edge_sparsity))
        # generate random edges with the edge_sparsity
        idx = np.random.randint(0, len(src_complete), n_edges)
        self.src = [src_complete[i] for i in idx]
        self.dst = [dst_complete[i] for i in idx]

        # buidl graph
        self.g = dgl.graph((self.src, self.dst), num_nodes=self.n_nodes)
        self.g = dgl.add_self_loop(self.g)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape= (1,1), dtype=np.float64)
        self.ob = self.start_idx
        self.step_count = 0

    def step(self, a):
        # Execute one time step within the environment
        # action: move from the current node to the node with index a; if there is an edge between these two nodes,
        # stays in teh current node
        done = False

        adj = self.g.adj().to_dense().numpy()

        if adj[self.ob, a] == 1:
            self.ob = a
        
        if self.ob == self.goal_idx:
            done = True
            r = 0
            return self.ob, r, done, {}
        else:
            r = -1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        
        return self.ob, r, done, {}

    def get_neighbor_nodes(self, idx):
        # get the neighbors indexes of the input node
        adj = self.g.adj().to_dense().numpy()
        indices = [i for i, x in enumerate(adj[idx,:]) if x == 1]
        return indices

    def reset(self, n_nodes = 20, edge_sparsity = 0.1, max_steps = 10, random_seed = None):
        self.n_nodes = n_nodes
        self.edge_sparsity = edge_sparsity
        self.max_steps = max_steps
        self.goal_idx = self.n_nodes - 1

        c_graph = nx.complete_graph(self.n_nodes)
        c_edges = list(c_graph.edges)
        src_complete = [item[0] for item in c_edges]
        dst_complete = [item[1] for item in c_edges]
        n_edges = int(np.ceil(len(src_complete) * self.edge_sparsity))
        # generate random edges with the edge_sparsity
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed()
        idx = np.random.randint(0, len(src_complete), n_edges)
        self.src = [src_complete[i] for i in idx]
        self.dst = [dst_complete[i] for i in idx]

        # build graph
        self.g = dgl.graph((self.src, self.dst), num_nodes=self.n_nodes)
        self.g = dgl.add_self_loop(self.g)

        self.ob = self.start_idx
        self.step_count = 0
        return self.ob

    def render(self, mode='human', close=False):
        pass
