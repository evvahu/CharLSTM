# split data without inlcuding unkn
# inlcude ukn when reading tokenising file 

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
from collections import defaultdict
from random import shuffle
import os
import gzip

#parser = argparse.ArgumentParser()
#parser.add_argument('--input', type=str, help='Input file path')
#parser.add_argument('--output', type=str, help='Output file path')
#parser.add_argument('--output_dir', type=str, help='Output path for training/valid/test sets')
#parser.add_argument('--vocab', type=int, default=10000, help="The size of vocabulary, default = 10K")

#args = parser.parse_args()
#logging.basicConfig(level=logging.INFO)


def read_gzip_stream(path):
    with gzip.open(path, 'rt', encoding="UTF-8") as f:
        for line in f:
            yield line

def read_text_stream(path):
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        with open(f_path, 'r', encoding="UTF-8") as f:
            for line in f:
                yield line
def read_file(path):
    if path.endswith(".gz"):
        logging.info("Reading GZIP file")
        return read_gzip_stream(path)
    else:
        return read_text_stream(path)


def create_vocab(path, vocab_size, lower=False):
    counter = defaultdict(int)
    for line in read_file(path):
        for word in line.replace("\n"," <eos>").split():
            if lower:
                counter[word.lower()] += 1
            else:
                counter[word] +=1

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:vocab_size]
    words = [w for (w, v) in count_pairs]
    print(len(count_pairs))
    print(len(counter), count_pairs[vocab_size - 1])
    w2idx = dict(zip(words, range(len(words))))
    idx2w = dict(zip(range(len(words)), words))
    return w2idx, idx2w


def create_char_vocab(path, vocab_size, lower=False):
    counter = defaultdict(int)
    for line in read_file(path):
        for word in line.strip().split():
            for cha in word:
                if lower:
                    counter[cha.lower()] += 1
                else:
                    counter[cha] +=1 

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:vocab_size]
    words = [w for (w, v) in count_pairs]
    print(len(count_pairs))
    print(len(counter), count_pairs[vocab_size - 1])
    ch2idx = dict(zip(words, range(len(words))))
    idx2ch = dict(zip(range(len(words)), words))
    return ch2idx, idx2ch
     
def convert_text(input_path, output_path, vocab):
    with open(output_path, 'w') as output:
        for line in read_file(input_path):
            words = [filter_word(word, vocab) for word in line.replace("\n", " <eos>").split()]
            output.write(" ".join(words) + "\n")
        output.close()

def convert_line(line, vocab, oov = False, lower=True):
    if oov:
        return [filter_word(word, vocab, lower) for word in line.replace("\n", " <eos>").split()]
    else:
        if lower:
            return[word.lower() for word in line.replace("\n", "<eos>").split()]
        else:
            return[word for word in line.replace("\n", "<eos>").split()]

def word_to_idx(word, vocab, lower):
    if lower:
        word = word.lower()
    if word in vocab:
        return vocab[word]
    else:
        return vocab["<unk>"]

def filter_word(word, vocab):
    if word in vocab:
        return word
    else:
        return "<unk>"

def create_corpus(input_path, output_path, vocab=dict(), oov=False):
    """ Split data to create training, validation and test corpus """
    nlines = 0
    f_train = open(output_path + "/train.txt", 'w')
    f_valid = open(output_path + "/valid.txt", 'w')
    f_test = open(output_path + "/test.txt", 'w')

    train = []

    for line in read_file(input_path):
        if nlines % 10 == 0:
            f_valid.write(" ".join(convert_line(line, vocab, oov)) + "\n")
        elif nlines % 10 == 1:
            f_test.write(" ".join(convert_line(line, vocab, oov)) + "\n")
        else:
            train.append(" ".join(convert_line(line, vocab, oov)) + "\n")
        nlines += 1

    shuffle(train)
    f_train.writelines(train)

    f_train.close()
    f_valid.close()
    f_test.close()

if __name__ == '__main__':
    input = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki/wiki_tiny'
    vocab = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy/vocab.txt'
    char_vocab = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy/char_vocab.txt'
    output = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy/output.txt'
    output_dir = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_no_unk_dummy'
    oov = False
    
    ch2idx, idx2ch = create_char_vocab(input, 50, lower=True) # 26, 10, 4, 5
    with open (char_vocab, 'w') as wf:
        for k,v in ch2idx.items():
            wf.write('{}\n'.format(k)) 

    w2idx, idx2w = create_vocab(input, 500, lower=True)
    with open (vocab, 'w') as wf:
        for k,v in w2idx.items():
            wf.write('{}\n'.format(k))
    
    create_corpus(input, output_dir, oov=False)
    #if oov:
    #    w2idx, idx2w = create_vocab(input, 50000)
    #    convert_text(input, output, w2idx)
    #    create_corpus(input, output_dir, w2idx)
    #else:
    #    create_corpus(input, output_dir)
    

