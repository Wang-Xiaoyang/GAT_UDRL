import numpy as np
import pandas as pd
from random import *

# dir_name = 'distance-first/'
# data = pd.read_pickle('./' + dir_name + 'sequential_samples_bert.pkl')

# actions to index
# '[PAD]': for 0 padding; '[CLS]': the start of sequences
# '[SEP]': seperate two sequences (if necessary)
# '[MASK]': masked tokens
def action2idx(n_actions):
    actions_list = [str(i) for i in list(range(n_actions))]
    word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
    for i, w in enumerate(actions_list):
        word2idx[w] = i + 4
    return word2idx


def fetch_list(original_list, token_len, start_idx):
    if start_idx + token_len > len(original_list):
        list_out = original_list[start_idx:].copy()
    else:
        list_out = original_list[start_idx:(start_idx+token_len)].copy()
    return list_out

# actions to tokens (tokens in word2idx)
# also get corresponding state sequences and rtg sequences
def get_splits(data, word2idx, token_len, in_each_action_seq):
    data_actions = data['labels']
    data_states = data['feats']
    data_rtg = data['returns']
    # in_each_action_seq: how many 'senteces' we select from each action sequence
    # token_len: the length of tokens, as input to BERT model
    token_list = list()
    states_list = list()
    rtg_list = list()
    action_seq_len = len(data_actions[0])
    for i in range(len(data_actions)):
        start_idx_i = np.random.randint(int(action_seq_len-token_len/2), size=in_each_action_seq)
        for j in start_idx_i:
            list_actions = fetch_list(data_actions[i], token_len, j)
            arr = [word2idx[str(s)] for s in list_actions]
            token_list.append(arr)
            # get corresponding states
            list_states = fetch_list(data_states[i], token_len, j)
            states_list.append(list_states)
            # get corresponding rtg
            list_rtg = fetch_list(data_rtg[i], token_len, j)
            rtg_list.append(list_rtg)
    return token_list, states_list, rtg_list


def prepare_data(token_list, states_list, rtg_list, word2idx, config):
    state_len = config.in_feats
    states_addi_info = {'[CLS]' : np.zeros(state_len),
                        '[SEP]' : np.ones(state_len)*-1.0}

    rtg_addi_info = {'[CLS]' : [0.0], '[SEP]' : [-1.0]}
    max_pred, maxlen = config.max_pred, config.maxlen
    vocab_size = len(word2idx)
    # read dataset
    batch = []
    for i in range(len(token_list)):
        input_ids = [word2idx['[CLS]']] + token_list[i] + [word2idx['[SEP]']]
        # MASK LM
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace

        # add '[CLS]' and '[SEP]' to states sequences and rtg sequences
        states = np.array([states_addi_info['[CLS]']] + list(states_list[i]) + [states_addi_info['[SEP]']])
        rtg = np.array(rtg_addi_info['[CLS]'] + list(rtg_list[i]) + rtg_addi_info['[SEP]'])

        # Zero Paddings
        n_pad = maxlen + 2 - len(input_ids) # maxlen plus [CLS] and [SEP]
        input_ids.extend([0] * n_pad)

        for _ in range(n_pad):
            states = np.append(states, np.zeros([1, state_len]), axis=0)

        rtg = list(rtg)
        rtg.extend([0] * n_pad)
        rtg = np.array(rtg)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        batch.append([states, rtg, input_ids, masked_tokens, masked_pos]) 
                                            # [seq of states (one episode), seq of rtg (one episode), masked actions, ...]

    return batch