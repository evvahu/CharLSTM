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
from data import Corpus

class Data(Dataset):
    def __init__(self, words, chars, nchars, seq_len):
        self.nchars = nchars
        self.seq_len = seq_len
        self.data_words, self.data_chars = self.batchify(words, chars)
        
    def batchify(self, data_words, data_chars):
        nseqs = data_words.size(0) // self.seq_len # floor division

        data_words = data_words.narrow(0,0,nseqs*self.seq_len)
        data_chars = data_chars.narrow(0,0, nseqs*self.seq_len)
        
        data_words = data_words.reshape(nseqs, self.seq_len)
        data_chars = data_chars.reshape(nseqs, self.seq_len*self.nchars)
        #assert data_words.shape[0] ==  int(data_chars.shape[0]/12), print('wrong batchifying')
        return data_words, data_chars

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_words.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return {'words': self.data_words[index], 'chars': self.data_chars[index]}


if __name__ == '__main__':
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy'
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/CharLSTMLM/testfiles'
    seq_len = 15
    nchars = 12
    corp = Corpus(path, nchars,seq_len)
    
    print(corp.train_words.shape, corp.train_chars.shape)
    ex_words = corp.train_words[:seq_len].tolist()
    ex_chars = corp.train_chars[0].tolist()
    print(ex_chars)

    word_list = [corp.dictionary.idx2word[int(id)] for id in ex_words]
    char_list = [corp.dictionary.idx2char[int(id)] for id in ex_chars]

    print(word_list)
    print(char_list)

    dat = Data(corp.train_words, corp.train_chars, nchars, seq_len)
    print(dat.data_words.shape, dat.data_chars.shape)
    
    ex_words = dat.data_words[-1].tolist()
    ex_chars = dat.data_chars[-1].tolist()

    word_list = [corp.dictionary.idx2word[int(id)] for id in ex_words]
    char_list = [corp.dictionary.idx2char[int(id)] for id in ex_chars]
    print(word_list)
    print(char_list) 