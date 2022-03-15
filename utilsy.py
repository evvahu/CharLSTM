# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import numpy as np
import utils_data

def batchify(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

def repackage_hidden(h):
    """Detaches hidden states from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_length):
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    # predict the sequences shifted by one word
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def char_repr(word, chardict, max_l):
    wchar = torch.zeros(max_l+1)
    eow = '<eow>'
    for i, ch in enumerate(word):
        ch = ch.lower()
        if i > max_l:
            return wchar
        wchar[i] = chardict[ch]
    if i == max_l:
        return wchar
    else:
        wchar[i+1] = chardict[eow]
        return wchar


def get_char_input(input, worddict, chardict, device, max_l=10):
    input = torch.flatten(input)
    char_mat = torch.empty(input.shape[0], max_l+1, dtype=int, device=device) #one word in each column
    for i, word in enumerate(input):
        char_word = char_repr(worddict[word], chardict, max_l)
        char_mat[i] = char_word
    return char_mat.T

def encode_sentence(sentence, worddict):
    torch.empty 

