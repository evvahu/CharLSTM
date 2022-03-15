# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from re import I
import torch
from collections import defaultdict
import logging
import numpy as np
import morph_analysis
import re 
from tqdm import tqdm
import multiprocessing as mp
from itertools import zip_longest

class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)
        vocab_path = os.path.join(path, 'vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))
        self.char2idx, self.idx2char, self.max_l = self.get_char_dict()
        print(self.char2idx)
        print(self.idx2char)


    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        #return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)

    def get_char_dict(self):
        char2idx = dict()
        idx2char = []
        i = 1
        max_l = 0
        for w in self.idx2word:
            if len(w) > max_l:
                max_l = len(w)
            for c in w:
                c = c.lower()
                if c not in char2idx:
                    idx2char.append(c)
                    char2idx[c] = i
                    i += 1
        
        char2idx['<eow>'] = i
        idx2char.append('<eow>')
        char2idx['<bow>'] = i + 1
        idx2char.append('<bow>')

        return char2idx, idx2char, max_l
   


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary(path)
        self.train = self.tokenize(self.dictionary, os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(self.dictionary, os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(self.dictionary, os.path.join(path, 'test.txt'))



    def tokenize(self, dictionary, path):
        """Tokenizes a text file for training or testing to a sequence of indices format
        We assume that training and test data has <eos> symbols """
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            ntokens = 0
            for line in f:
                words = line.split()
                ntokens += len(words)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(ntokens)
            token = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word in dictionary.word2idx:
                        ids[token] = dictionary.word2idx[word]
                    else:
                        ids[token] = dictionary.word2idx["<unk>"]
                    token += 1

        return ids

    