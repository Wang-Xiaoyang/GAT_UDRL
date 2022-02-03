#!/usr/bin/env python
# coding: utf-8

# implement a BERT model
# Can be further used for GAT_UDRL
# Example from: https://github.com/wmathor/nlp-tutorial/blob/master/5-2.BERT/BERT-Torch.ipynb

# BERT model with UDRL.
# Task: predicting masked actions - 'masked action model (MAM)'


'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, modify by wmathor
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
         https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
'''
import re
import math
import torch
import numpy as np
import pandas as pd
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import wandb
import time

from bert_udrl_utils import *
from BERT_model import *

wandb.init(entity="xiaoyang",
                project="BERT_UDRL")

wandb.config.n_nodes = 20
wandb.config.n_classes = 2
wandb.config.model = 'BERT_UDRL'
wandb.config.task = 'node_association'
wandb.config.offline_data = 'distance-first'

wandb.config.n_epoches = 200
wandb.config.batch_size = 16
wandb.config.learning_rate = 0.001
wandb.config.training_set = 0.8
# here 3 means [value (1*1), location (x,y) (1*2)]
wandb.config.in_feats = 3 + wandb.config.n_classes*2

# BERT parameters
wandb.config.maxlen = 10 # the max token length (max length for trajectories)
wandb.config.in_each_action_seq = 2
wandb.config.max_pred = 5 # max tokens of prediction
wandb.config.n_layers = 6
wandb.config.n_heads = 12
wandb.config.d_model = 768
wandb.config.d_ff = 768*4 # 4*d_model, FeedForward dimension
wandb.config.d_k = wandb.config.d_v = 64  # dimension of K(=Q), V
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t0 = time.time()

# read dataset
dir_name = 'distance-first/'
data = pd.read_pickle('./' + dir_name + 'sequential_samples_bert.pkl')

# get vocabularies
word2idx = action2idx(config.n_classes)
vocab_size = len(word2idx)

token_list, states_list, rtg_list = get_splits(data, word2idx, config.maxlen, config.in_each_action_seq)

prepared_data = prepare_data(token_list, states_list, rtg_list, word2idx, config)
# Proprecessing Finished
print('time for processing dataset with %d samples is %f' % (len(data), time.time() - t0))

# build dataset
class MyDataSet(Data.Dataset):
    def __init__(self, prepared_data):
        states_seq, rtg_seq, input_ids, masked_tokens, masked_pos = zip(*prepared_data)
        states_seq, rtg_seq, input_ids, masked_tokens, masked_pos = torch.FloatTensor(states_seq), torch.FloatTensor(rtg_seq).unsqueeze(2), \
                                        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos)

        self.states_seq = states_seq
        self.rtg_seq = rtg_seq
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
  
    def __len__(self):
        return len(self.input_ids)
  
    def __getitem__(self, idx):
        return self.states_seq[idx], self.rtg_seq[idx], self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]

# training set and testing set
# split: 80%, 20%
num_samples = len(prepared_data)
num_train = int(num_samples * config.training_set)
batch_size = config.batch_size

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_samples))

train_dataloader = Data.DataLoader(
    MyDataSet(prepared_data),
    sampler=train_sampler,
    batch_size=batch_size,
    drop_last=False)

test_dataloader = Data.DataLoader(
    MyDataSet(prepared_data),
    sampler=test_sampler,
    batch_size=batch_size,
    drop_last=False)

# model and training

model = BERT(config, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

for epoch in range(config.n_epoches):
    loss_sum = .0
    num_batch = 0
    for states_seq, rtg_seq, input_ids, masked_tokens, masked_pos in train_dataloader:
        states_seq, rtg_seq = states_seq.to(device), rtg_seq.to(device)
        input_ids, masked_pos = input_ids.to(device), masked_pos.to(device)
        masked_tokens = masked_tokens.to(device)

        logits_lm = model(states_seq, rtg_seq, input_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss = loss_lm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss
        num_batch += 1

    wandb.log({"average loss in epoches": loss_sum/num_batch})

    if (epoch + 1) % 20 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss_sum/num_batch))

# test
pred_true = 0
total_test = 0
pred_true_actual_actions = 0
total_masked_actions = 0
for states_seq, rtg_seq, input_ids, masked_tokens, masked_pos in test_dataloader:
    states_seq, rtg_seq = states_seq.to(device), rtg_seq.to(device)
    input_ids, masked_pos = input_ids.to(device), masked_pos.to(device)
    masked_tokens = masked_tokens.to(device)

    logits_lm = model(states_seq, rtg_seq, input_ids, masked_pos)

    preds = torch.argmax(logits_lm.view(-1, vocab_size), dim=1)

    pred_true += (preds == masked_tokens.view(-1)).sum()
    total_test += len(masked_tokens.view(-1))

    for i in range(len(masked_tokens.view(-1))):
        if masked_tokens.view(-1)[i] != 0:
            pred_true_actual_actions += (preds[i] == masked_tokens.view(-1)[i])
            total_masked_actions += 1

wandb.log({"Action Prediction Accuracy": 
            pred_true_actual_actions / total_masked_actions})

print("Total Accuracy: %.3f " % (pred_true / total_test))
print("Action Prediction Accuracy: %.3f ; %d / %d" % (pred_true_actual_actions \
    / total_masked_actions, pred_true_actual_actions, total_masked_actions))