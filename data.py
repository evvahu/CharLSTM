# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from email.policy import default
import os
from re import I
from regex import W
import torch
from collections import defaultdict
import logging
import numpy as np
import re 
from tqdm import tqdm
import multiprocessing as mp
from itertools import zip_longest
from multiprocessing import Pool
from itertools import repeat
import math 

class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)
        self.char2freq = defaultdict(int)
        self.char2idx = {}
        self.idx2char = []
        vocab_path = os.path.join(path, 'vocab.txt')
        char_vocab_path = os.path.join(path, 'char_vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            char_vocab = open(char_vocab_path, encoding='utf8').read()
            self.char2idx = {c:i+1 for i,c in enumerate(char_vocab.split())}
            self.idx2char = [c for c in char_vocab.split()]
            self.char2idx['*PAD*'] = 0
            self.idx2char.insert(0, '*PAD*')
            
            self.add_char('<unk>')
            self.add_char('<bow>')
            self.add_char('<eow>')
            self.idx2word = [w for w in vocab.split()]
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.add_word('<unk>')
            
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, create a vocab file first.")

    def add_char(self, char):
        self.char2freq[char] += 1
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

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
        idx2char = ['*PAD*']
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
    def __init__(self, path, word_max_l, seq_l, cpu_count=None):
        self.max_l = word_max_l
        self.seq_l = seq_l
        if not cpu_count:
            self.cpu_count = mp.cpu_count()
        else:
            self.cpu_count = cpu_count
        self.dictionary = Dictionary(path)
        self.train_words, self.train_chars, self.train_targets = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_words, self.valid_chars, self.valid_targets = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_words, self.test_chars, self.test_targets = self.tokenize(os.path.join(path, 'test.txt'))

    def _grouper(self, n, iterable, padvalue=None):
        return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


    def parallel_tokenize(self, chunk):
        words = chunk.split()
        ntokens = len(words)
        ids = torch.zeros(ntokens, dtype=torch.long)
        ids_chars = torch.zeros((ntokens, self.max_l))
        ids_chars_target = torch.zeros((ntokens, self.max_l))
        for i, w in enumerate(words):
            if w in self.dictionary.word2idx:
                ids[i] = self.dictionary.word2idx[w]
            else:
                ids[i] = self.dictionary.word2idx["<unk>"] 
            char_tens = torch.zeros(self.max_l)
            char_tens_t = torch.zeros(self.max_l)
            c_local = 0
            too_long = False
            if len(w) > (self.max_l-2):
                w = w[:(self.max_l-2)]
            for c in w: 
                if c in self.dictionary.char2idx:
                    char_tens[c_local] = self.dictionary.char2idx[c]
                    char_tens_t[c_local+1] = self.dictionary.char2idx[c]
                else:
                    char_tens[c_local] = self.dictionary.char2idx['<unk>']
                    char_tens_t[c_local+1] = self.dictionary.char2idx['<unk>']
                c_local +=1 

            char_tens[c_local] = self.dictionary.char2idx['<eow>']
            char_tens_t[0] = self.dictionary.char2idx['<bow>']

            ids_chars[i:, ] = char_tens
            ids_chars_target[i:,] = char_tens_t

        return ids, ids_chars, ids_chars_target

    def tokenize(self, path):
        """Tokenizes a text file for training or testing to a sequence of indices format
        We assume that training and test data has <eos> symbols """
        assert os.path.exists(path)
        # Tokenize file content
        reader = open(path, 'r', encoding="utf8")
        pool = mp.Pool(self.cpu_count)
        words = torch.zeros(0)
        chars = torch.zeros(0)
        target = torch.zeros(0)
        for chunk in self._grouper(10, reader):
            res = pool.map(self.parallel_tokenize, chunk)

            for tup in res:
                words = torch.cat((words, tup[0].float()))
                chars = torch.cat((chars, tup[1]))
                target = torch.cat((chars, tup[2]))
        pool.close()
        return words, chars, target

if __name__ == '__main__':
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy'
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/CharLSTMLM/testfiles'
    corp = Corpus(path, 12, 15, 1)