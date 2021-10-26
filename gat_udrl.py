import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import dgl.data
from dgl.dataloading import GraphDataLoader
from gat_udrl_utils import *

import time
import wandb

wandb.init(entity="xiaoyang",
                project="GAT_UDRL")

#env config
wandb.config.n_nodes = 20
wandb.config.n_classes = 4
wandb.config.model = 'GAT_UDRL_1'

wandb.config.n_epoches = 500
wandb.config.batch_size = 16
wandb.config.learning_rate = 0.01
wandb.config.training_set = 0.8
wandb.config.in_feats = 7
wandb.config.out_feats_GAT = 16
wandb.config.out_feats_rtg_encoder = 8
wandb.config.num_GAT_heads = 3
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()
dataset = GAT_Dataset()
print('time for processing dataset is %f' % (time.time() - t0))

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
out_feats = config.n_nodes # number of classes
learning_rate = config.learning_rate
epochs = config.n_epoches

model = GAT(in_feats, out_feats_gat, num_heads, h_feats, out_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

wandb.watch(model)

for epoch in range(epochs):
    losses = 0
    num_batches = 0
    for batched_graph, labels, returns, cur_items in train_dataloader:
        # cur_items = torch.FloatTensor(cur_items)
        cur_items = torch.stack(cur_items, dim=1).float()
        batched_graph, labels, returns, cur_items  = batched_graph.to(device), labels.to(device), returns.to(device), cur_items.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'].float(), returns.float(), cur_items)
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss
        num_batches += 1
    
    wandb.log({"average loss in epoches": losses/num_batches, "epoch": epoch})    

num_correct = 0
num_tests = 0
for batched_graph, labels, returns, cur_items in test_dataloader:
    cur_items = torch.FloatTensor(cur_items)
    batched_graph, labels, returns, cur_items  = batched_graph.to(device), labels.to(device), returns.to(device), cur_items.to(device)
    pred = model(batched_graph, batched_graph.ndata['feat'].float(), returns.float(), cur_items)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

wandb.log({"test accuracy": num_correct / num_tests})