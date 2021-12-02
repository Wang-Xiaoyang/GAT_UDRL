import numpy as np
import dgl
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import dgl.data
from dgl.dataloading import GraphDataLoader
from shortest_path_utils import *
from GATModel import *
from env_utils import *

import time
import gym
import wandb

torch.cuda.empty_cache()

wandb.init(entity="xiaoyang",
                project="GAT_UDRL")

#env config
"""
GAT_UDRL_2: 
Node feature for graph: [State(node idx), action taken in for this node (for previous nodes), 
    previous return-to-go]
graph with node features [s, previous actions, previous rtg] --> GAT --> node embedding  --> prediction layer
"""
wandb.config.is_training = True
# the model name to load
wandb.config.run_name = ''

wandb.config.n_nodes = 20
wandb.config.n_classes = 20
wandb.config.model = 'GAT_UDRL_2'
wandb.config.offline_data = 'shortest-path-bfs' # shortest-path-bfs; shortest-path-random-walk
wandb.config.edge_sparsity = 0.1

wandb.config.n_epoches = 2000
wandb.config.batch_size = 64
wandb.config.learning_rate = 0.005
wandb.config.training_set = 0.8
# wandb.config.in_feats = 1 + wandb.config.n_classes
# wandb.config.in_feats = wandb.config.n_nodes + wandb.config.n_classes + 1 + 1 # node index: one-hot encoder; +1 means return; another +1 is the goal index indicator
wandb.config.in_feats = 4
wandb.config.out_feats_GAT = 16
wandb.config.out_feats_rtg_encoder = 20
wandb.config.num_GAT_heads = 3
wandb.config.evaluation_eps = 100
wandb.config.freeze_reward_encoder = False
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()
dataset = GAT_Dataset_2()
print('time for processing dataset with %d samples is %f' % (len(dataset), time.time() - t0))

# training set and testing set
num_examples = len(dataset)
num_train = int(num_examples * config.training_set)
batch_size = config.batch_size

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset,
    sampler=train_sampler,
    batch_size=batch_size,
    drop_last=False)

test_dataloader = GraphDataLoader(
    dataset,
    sampler=test_sampler,
    batch_size=batch_size,
    drop_last=False)

# Create the model
in_feats = config.in_feats # the in_feats of nodes of graphs
out_feats_gat = config.out_feats_GAT # the output dimension of GATs
h_feats = config.out_feats_rtg_encoder # the output dimension of the reward-to-go encoder
num_heads = config.num_GAT_heads # num of GAT heads
out_feats = config.n_classes # number of classes
learning_rate = config.learning_rate
epochs = config.n_epoches

model = GAT2(in_feats, out_feats_gat, num_heads, h_feats, out_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# freeze the reward-to-go encoding layer
if config.freeze_reward_encoder == True:
    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False

wandb.watch(model)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000, 3000, 4500], gamma=0.5)

if config.is_training == True:
    for epoch in range(epochs):
        losses = 0
        num_batches = 0
        for batched_graph, labels, returns, cur_items in train_dataloader:
            # cur_items = torch.FloatTensor(cur_items)
            cur_items = torch.stack(cur_items, dim=1).float()
            batched_graph, labels, returns, cur_items  = batched_graph.to(device), labels.to(device), returns.to(device), cur_items.to(device)
            pred = model(batched_graph, batched_graph.ndata['feat_with_returns'].float(), cur_items)
            # remove action information
            # pred = model(batched_graph, batched_graph.ndata['feat'][:,:-config.n_classes].float(), returns.float(), cur_items)
            loss = F.cross_entropy(pred, labels, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss
            num_batches += 1
        
        wandb.log({"average loss in epoches": losses/num_batches})
        scheduler.step()  

    # save model
    PATH = './model_shortest_path/model_' + wandb.run.name + '.pt'
    torch.save(model.state_dict(), PATH)

    # accuracy of trained model
    num_correct = 0
    num_tests = 0
    for batched_graph, labels, returns, cur_items in test_dataloader:
        cur_items = torch.stack(cur_items, dim=1).float()
        batched_graph, labels, returns, cur_items  = batched_graph.to(device), labels.to(device), returns.to(device), cur_items.to(device)
        # pred = model(batched_graph, batched_graph.ndata['feat'][:,:-config.n_classes].float(), returns.float(), cur_items)
        pred = model(batched_graph, batched_graph.ndata['feat_with_returns'].float(), cur_items)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    wandb.log({"test accuracy": num_correct / num_tests})

    # the accuracy on training set?
    num_correct = 0
    num_tests = 0
    for batched_graph, labels, returns, cur_items in train_dataloader:
        cur_items = torch.stack(cur_items, dim=1).float()
        batched_graph, labels, returns, cur_items  = batched_graph.to(device), labels.to(device), returns.to(device), cur_items.to(device)
        # pred = model(batched_graph, batched_graph.ndata['feat'][:,:-config.n_classes].float(), returns.float(), cur_items)
        pred = model(batched_graph, batched_graph.ndata['feat_with_returns'].float(), cur_items)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    wandb.log({"accuracy on training set": num_correct / num_tests})
    print('-------------------Model training finished-------------------')


else:
    PATH = './model_shortest_path/model_' + config.run_name + '.pt'  
    model.load_state_dict(torch.load(PATH))


# evaluation and baselines
env = gym.make('gym_graph_shortest_path:graph_shortest_path-v0')
env.reset(n_nodes = config.n_nodes, edge_sparsity=config.edge_sparsity)
n_actions = env.action_space.n

# generate seed list
np.random.seed(10)
seed_list = np.random.randint(1, high=1e5, size=config.evaluation_eps)

"""returns
Reference result: breadth-first-search agent.
"""
reward_list_bfs = []
for idx in range(config.evaluation_eps):
    done = False
    env.reset(n_nodes = config.n_nodes, edge_sparsity = config.edge_sparsity, random_seed = seed_list[idx])
    reward = 0
    # bfs
    _, actions_bfs = bfs_shortest_path(env.start_idx, env.goal_idx, env)
    actions_bfs = actions_bfs[1:]
    while not done:
        # breadth-first-search
        if len(actions_bfs) == 0:
            a = env.ob
        else:
            a = actions_bfs[0]
            actions_bfs.pop(0)
            
        _, r, done, _ = env.step(a)
        reward += r 
    wandb.log({"reference reward - breadth-first-search agents": reward})
    reward_list_bfs.append(reward)

"""
model evaluation on env
"""
for idx in range(config.evaluation_eps):
    done = False
    # use known seeds in test time; compare between trained model and random model
    env.reset(n_nodes = config.n_nodes, edge_sparsity = config.edge_sparsity, random_seed = seed_list[idx])
    # ndata in graph
    feats = np.array(list(range(config.n_nodes))) / config.n_nodes # normalized to 1
    feats = np.expand_dims(feats, axis=1)
    nodes_actions = np.zeros((config.n_nodes, 1))
    node_goal = np.zeros((config.n_nodes, 1))
    node_goal[env.goal_idx] = 1
    node_goal[env.start_idx] = -1
    actions = []
    rewards = []
    cur_nodes = []
    returns = -1.0
    # returns = reward_list_bfs[idx] #bfs guided search
    returns_all = np.zeros(int(config.n_nodes))
    total_reward = 0
    while not done:
        feats_with_actions = np.concatenate((feats, node_goal, nodes_actions), axis=1)
        # a one-hot vector representing the current node, corresponding with labels
        current_node = np.zeros(env.n_nodes)
        current_node[env.ob] = 1
        current_node = torch.Tensor(current_node).to(device)
        # current graph
        g = dgl.graph((env.src, env.dst), num_nodes=config.n_nodes)
        g = dgl.to_bidirected(g)
        g = dgl.add_self_loop(g)

        returns_all[env.ob] = returns/10 # normalisation
        feat_with_returns = torch.from_numpy(np.concatenate((feats_with_actions, np.expand_dims(returns_all, axis=1)), axis=1))
        g.ndata['feat_with_returns'] = feat_with_returns
        g = g.to(device)
        pred = model(g, g.ndata['feat_with_returns'].float(), current_node.unsqueeze(0))
        a = pred.argmax().cpu().numpy()
        # encode the action taken
        nodes_actions[env.ob] = a / env.action_space.n 
        _, r, done, _ = env.step(a)
        returns -= r
        if returns > 0.0:
            returns = 0.0
        total_reward += r
    wandb.log({"evaluation: trained model rewards (each episode)": total_reward})

"""
Reference result: returns in random agent.
"""
for idx in range(config.evaluation_eps):
    done = False
    env.reset(n_nodes = config.n_nodes, edge_sparsity = config.edge_sparsity, random_seed = seed_list[idx])
    reward = 0
    while not done:
        a = env.action_space.sample()
        _, r, done, _ = env.step(a)
        reward += r     
    wandb.log({"reference - reward of random agents": reward})

"""
Reference result: random-walk agent.
"""
for idx in range(config.evaluation_eps):
    done = False
    env.reset(n_nodes = config.n_nodes, edge_sparsity = config.edge_sparsity, random_seed = seed_list[idx])
    reward = 0
    while not done:
        # random walk
        a_list = env.get_neighbor_nodes(env.ob)
        # if there are other choices, avoid staying at current node
        if len(a_list) > 1:
            a_list.remove(env.ob)
        a = random.choice(a_list)
        _, r, done, _ = env.step(a)
        reward += r 
    wandb.log({"reference reward - random-walk agents": reward})