from torch.utils.data import DataLoader, Dataset
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


class Data(Dataset):
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        return X, y


class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []
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
        print(self.char2idx)
        print(self.idx2char)

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word), len(self.idx2char)

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