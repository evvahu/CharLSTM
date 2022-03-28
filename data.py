# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from email.policy import default
import os
from re import I
import torch
from collections import defaultdict
import logging
import numpy as np
import re 
from tqdm import tqdm
import multiprocessing as mp
from itertools import zip_longest

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
            self.word2idx = {w: i for i, w in enumerate(vocab.split())}
            self.add_word('<unk>')
            self.idx2word = [w for w in vocab.split()]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, create a vocab file first.")
            #self.create_vocab(os.path.join(path, 'train.txt'))
            #open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))
        #self.char2idx, self.idx2char, self.max_l = self.get_char_dict()
        print(self.char2idx)
        print(self.idx2char)

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
    def __init__(self, path, word_max_l):
        self.dictionary = Dictionary(path)
        print('finished')
        self.train_words, self.train_chars = tokenize(self.dictionary, os.path.join(path, 'train.txt'), word_max_l)
        print('finished loading train')
        self.valid_words, self.valid_chars = tokenize(self.dictionary, os.path.join(path, 'valid.txt'), word_max_l)
        self.test_words, self.test_chars = tokenize(self.dictionary, os.path.join(path, 'test.txt'), word_max_l)
        print('finished tokenising all data files')


def tokenize(dictionary, path, max_l):
    """Tokenizes a text file for training or testing to a sequence of indices format
       We assume that training and test data has <eos> symbols """
    assert os.path.exists(path)
    nr_lines = 0
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            nr_lines +=1
            words = line.split()
            ntokens += len(words)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        ids = torch.LongTensor(ntokens)
        ids_chars = torch.LongTensor(ntokens, max_l)
        token = 0
        for line in tqdm(f, total=nr_lines):
        #for line in f:
            line = line.strip()
            if not line: continue
            words = line.split()
            for word in words:
                if word in dictionary.word2idx:
                    ids[token] = dictionary.word2idx[word]
                else:
                    ids[token] = dictionary.word2idx["<unk>"]
                
                char_tens = torch.zeros(max_l, dtype=int)
                c_local = 0
                too_long = False
                for c in word: 
                    if c in dictionary.char2idx:
                        char_tens[c_local] = dictionary.char2idx[c]
                    else:
                        char_tens[c_local] = dictionary.char2idx['<unk>']
                    c_local +=1 
                    if c_local == (max_l-1):
                        char_tens[c_local] = dictionary.char2idx['<eow>']
                        too_long = True
                        break
                if not too_long:
                    char_tens[c_local] = dictionary.char2idx['<eow>']
                #ids_chars = torch.cat((ids_chars, char_tens))
                ids_chars[token,:] = char_tens
                token += 1
    return ids, ids_chars

if __name__ == '__main__':
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy'
    corp = Corpus(path, 12)
