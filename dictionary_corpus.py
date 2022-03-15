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

class AllDictionaries(object):
    def __init__(self, path):
        self.path_words = os.path.join(path, 'vocab.txt')
        self.path_morphs = os.path.join(path, 'morph_vocab.txt') 
        self.path_chars = os.path.join(path, 'char_vocab.txt')

        word_vocab = open(self.path_words, encoding="utf8").read()
        self.word2idx = {w: i for i, w in enumerate(word_vocab.split())}
        self.idx2word = [w for w in word_vocab.split()]

        morph_vocab = open(self.path_morphs, encoding="utf8").read()
        self.morph2idx = {w: i for i, w in enumerate(morph_vocab.split())}
        self.idx2morph = [w for w in morph_vocab.split()]

        char_vocab = open(self.path_chars, encoding="utf8").read()
        self.char2idx = {w: i for i, w in enumerate(char_vocab.split())} 
        self.idx2char = [w for w in char_vocab.split()] 


class Dictionary(object):
    def __init__(self, path, mtags_path):
        self.word2idx = {}
        self.idx2word = []
        self.word2freq = defaultdict(int)
        self.morphtags_path = mtags_path
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
        self.morph2idx, self.idx2morph = self.get_morph_dict()
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
        char2idx['</w>'] = i
        idx2char.append('</w>')
        return char2idx, idx2char, max_l
    def get_morph_dict(self):
        morph2idx = dict()
        idx2morph = []
        i = 0
        with open(self.morphtags_path, 'r') as rf:
            for l in rf:
                l = l.strip()
                if l not in morph2idx:
                    morph2idx[i] = l
                    i+=1
                    idx2morph.append(l)
        return morph2idx, idx2morph

class Corpus(object):
    def __init__(self, path, morph_vocab = ""):#, morph_path):
        #self.dictionary = Dictionary(path, morph_path)
        #self.test = tokenize(self.dictionary, os.path.join(path, 'test.txt'))
        self.vocab_file_exists = True
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
        char_path = os.path.join(path, 'char_vocab.txt')
        morph_path = os.path.join(path, 'morph_vocab.txt')
        if os.path.isfile(char_path):
            self.morph_file_exists = True
        else:
            self.morph_file_exists = False
        if os.path.isfile(char_path):
            self.char_file_exists = True 
        else:
            self.char_file_exists = False

        self.train = self.tokenize_morph_char(path, os.path.join(path, 'train.txt'), self.word2idx)
        print('done with train')
        if os.path.isfile(char_path):
            self.morph_file_exists = True
        else:
            self.morph_file_exists = False
        if os.path.isfile(char_path):
            self.char_file_exists = True 
        else:
            self.char_file_exists = False
        self.valid = self.tokenize_morph_char(path, os.path.join(path, 'valid.txt'),  self.word2idx)
        self.test = self.tokenize_morph_char(path, os.path.join(path, 'test.txt'),  self.word2idx)


    def edit_morph(self, morph):
        try:
            form , morph = morph.split('/', maxsplit=1)
            if '/' in morph:
                morph = morph.split('/')
            else:
                morph = [morph]
            morph = [m.replace('<', ' <') for m in morph]
            morph_fin = []
            for m in morph:
                if not m.endswith('<lower>'):
                    morph_fin.append(m)
            return morph_fin
        except:
            return ""

            
    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)
    def tokenize_morph_char(self, path_root, path, word_dict, lang_morph = 'deu_morph'):
        assert os.path.exists(path)
        all_words = []
        nmorphs = 0
        ntokens = 0
        nchars = 0
        vocab_char = set()
        vocab_morph = set()
        if not self.morph_file_exists:
            wf_morph = open(os.path.join(path_root, 'morph_vocab.txt'), 'w', encoding='utf8')
        if not self.char_file_exists:
            wf_char = open(os.path.join(path_root, 'char_vocab.txt'),'w', encoding='utf8')
        num_lines = sum(1 for line in open(path,'r'))
        with open(path, 'r', encoding="utf8") as f:
            for line in tqdm(f, total=num_lines):
                line = line.strip()
                if not line: continue
                w_list = []
                m_list = []
                c_list = []
                words = line.split()
                for w in words:
                    if w in word_dict:
                        w_list.append(w)
                    else:
                        w_list.append('<unk>')
                    c_cur = []
                    for c in w:
                        c_cur.append(c)
                        if c not in vocab_char:
                            vocab_char.add(c)
                            if not self.char_file_exists:
                                wf_char.write('{}\n'.format(c))
                            

                    nchars += len(c_cur)
                    c_list.append(c_cur)
                line = line.replace("\"", "")
                morph = morph_analysis.get_morph_analysis(line)
                #print(morph)
                #morph = morph.strip().strip('q^')
                morph = morph.split(' ')
                for morp in morph:
                    morp = morp.strip().strip('$')
                    m_list.append(self.edit_morph(morp))
                nmorphs += len(morph)
                ntokens += len(words)
                for wo, mo, cha in zip(w_list, m_list[:-1], c_list):
                    if wo == '<unk>':
                        if len(mo) > 0:
                            new_m = []
                            if mo[0].startswith('*'):
                                new_m = ['<unk>']
                            else: 
                                for m in mo:
                                    if ' ' in m:
                                        _, m = m.split(' ', maxsplit=1)
                                        m = '<unk> ' + ''.join(m)
                                        new_m.append(m)
                                    else:
                                        new_m.append('<unk>')
                            mo = new_m
                    if not self.morph_file_exists:
                        for m in mo:
                            if ' ' in m:
                                tags = m.split(' ')
                                for t in tags:
                                    if t not in vocab_morph:
                                        wf_morph.write('{}\n'.format(t))
                                        vocab_morph.add(t)
                            else:
                                if t not in vocab_morph:
                                    wf_morph.write('{}\n'.format(m))
                                    vocab_morph.add(t)

                    all_words.append((wo, mo, cha))

        return all_words






def grouper(n, iterable, padvalue=None):
	"""grouper(3, 'abcdefg', 'x') -->
	('a','b','c'), ('d','e','f'), ('g','x','x')"""

	return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parallel_tokenise(line):
    nmorphs = 0
    ntokens = 0
    nchars = 0
    vocab_char = set()
    vocab_morph = set()
    sent = []
    #wf_morph = open(os.path.join(path_root, 'morph_vocab.txt'), 'a', encoding='utf8')
   # wf_char = open(os.path.join(path_root, 'char_vocab.txt'),'w', encoding='utf8')
    line = line.strip()
    line = line.replace("\"", "")
    line = line.replace('(', '')
    line = line.replace(')', '')
    #line = line.strip('\"')
    w_list = []
    m_list = []
    c_list = []
    words = line.split()
    for w in words:
        w_list.append(w)
        c_cur = []
        for c in w:
            c_cur.append(c)
            vocab_char.add(c)
        nchars += len(c_cur)
        c_list.append(c_cur)
        try:
            morph = morph_analysis.get_morph_analysis(line)
        except:
            continue
        morph = morph.split(' ')
        for morp in morph:
            morp = morp.strip().strip('$')
            m_list.append(edit2_morph(morp))
        nmorphs += len(morph)
        ntokens += len(words)
    for wo, mo, cha in zip(w_list, m_list[:-1], c_list):
        if wo == '<unk>':
            if len(mo) > 0:
                new_m = []
                if mo[0].startswith('*'):
                    new_m = ['<unk>']
                else: 
                    for m in mo:
                        if ' ' in m:
                            _, m = m.split(' ', maxsplit=1)
                            m = '<unk> ' + ''.join(m)
                            new_m.append(m)
                        else:
                            new_m.append('<unk>')
                mo = new_m

        for m in mo:
            if ' ' in m:
                tags = m.split(' ')
                for t in tags:
                    vocab_morph.add(t)
            else:
                vocab_morph.add(m)

        sent.append((wo, mo,cha))
    return sent

def edit2_morph(morph):
    try:
        form , morph = morph.split('/', maxsplit=1)
        if '/' in morph:
            morph = morph.split('/')
        else:
            morph = [morph]
        morph = [m.replace('<', ' <') for m in morph]
        morph_fin = []
        for m in morph:
            if not m.endswith('<lower>'):
                morph_fin.append(m)
        return morph_fin
    except:
        return ""

if __name__ == "__main__":
    path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/BASQUE/bmc/bmc_sents.txt'
    #path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/test_data/dummy.txt'
    print("Number of processors: ", mp.cpu_count())
    outpath = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/Basque/out_charmorph.txt'
    #pool = mp.Pool(mp.cpu_count())
    p = mp.Pool(mp.cpu_count())
    #p = mp.Pool(8)
    reader = open(path, 'r')
    with open(outpath, 'w') as wf:
        for chunk in grouper(1000, reader):
            res = p.map(parallel_tokenise, chunk)
            print('chunk done')
            for r in res:
                for s in r:
                    wf.write('{}\t{}\t{}\n'.format(s[0], s[1], s[2]))

    #corp = Corpus(path)
    #char_path = os.path.join(path, 'vocab_morph.txt')
    #print(corp.train, corp.test)
    #print(corp.word2idx)
    #all_words = tokenize_morph_char(path, corp.word2idx)


    #https://gist.github.com/ngcrawford/2237170