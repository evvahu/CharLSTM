# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import numpy as np
import utils_data

def batchify(data, bsz):#, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if cuda:
    #    data = data.cuda()
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

def char_repr(word, chardict, max_l, eow):
    wchar = torch.zeros(max_l+1, dtype=int)
    #eow = '<eow>'
    for i, ch in enumerate(word):
        ch = ch.lower()
        if i+1 > max_l:
            #print('s1')
            wchar[i] = eow
            return wchar

        wchar[i] = chardict[ch]
    #if i+1 == max_l:
     #   print('s2')
      #  wchar[i+1] = chardict[eow]
       # return wchar, length
    #else:
    #print('s3')
    wchar[i+1] = eow
    return wchar


def get_char_input(input, dictionary, eow, max_l=10):
    
    #print(input)
    char_mat = torch.empty(max_l+1,input.shape[1]*input.shape[0], dtype=int)
    input = input.T.flatten(0)
    #char_mat = torch.empty(input.shape[0], max_l+1, dtype=int, device=device) #one word in each column
    for i, word in enumerate(input):
        #print(worddict[word])
        char_word = char_repr(dictionary.idx2word[word], dictionary.char2idx, max_l, eow)
        char_mat[:, i] = char_word
    return char_mat


