
import argparse
from data import Corpus, Dictionary
from data_loader import Data
from utilsy import batchify, get_batch, get_char_input, repackage_hidden, get_char_batch
from models import Encoder, CharGenerator
import torch
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
import logging
import time 
import toml 
import os 
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(data):
    """
    see train() for analogous code 
    """
    model.eval()
    avg_prob = []
    avg_seq_loss = []
    hidden_state = model.init_hidden(config['bs'])
    hidden_char = model.charEncoder.init_hidden(config['bs'])
    hidden_generator = generator.init_hidden(1) # only one word at a time 
    with torch.no_grad():
        for batch_ndx, sample in enumerate(data):
            # get batch for word and character data
            data_word = sample['words']
            data_char = sample['chars']

            beginning_char = 0
            end_char = config['word_length']
            seq_loss = 0
            # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
            for id in range(data_word.shape[1]-1): # -1 because after final word no word to predict 
                # get the correct characters for word 'id'
                word_current = data_word[:,id].long()
                chars_current = data_char[:,beginning_char:end_char].long()
                chars_target = data_char[:,end_char:(end_char+config['word_length'])].long()
                #print('SHAPESS', word_current.shape, chars_target.shape, chars_current.shape)
                #data_char_target_word = target_char[:,end_char:end_char+config['bs']] 
                if torch.cuda.is_available():
                    word_current.cuda()
                    chars_current.cuda()
                # send word, characters to main LSTM forward call
                if word_current.shape[0] != config['bs']:
                    break
                output, hidden_state, hidden_char = model(word_current, chars_current, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch    
                # generate next word based on hidden_state of LSTM with generator 
                #lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
                loss = 0
                #nr_words = hidden_state[0].shape[1]-1
                nr_words = word_current.shape[0]
                #print(hidden_state[0].shape)
                for word_nr in range(nr_words-1): # 34 
                    #wl = lengths[word_nr]
                    char_input = chars_current[word_nr].unsqueeze(0)
                    #print('char input', char_input.shape)
                    hs = hidden_state[0][1, word_nr, :] #[:, word_nr] # 1 x hidden size
                    #hs = hidden_state[0]
                    t = chars_target[word_nr] # t is of size word_length, padded with 0s at the moment
                    stringy = [corpus.dictionary.idx2char[ti] for ti in t]
                    # words contain <eow> token at the end of word (final char)
                    l, probs,outcome= generate_word(hs, hidden_generator, t, eow,config['word_length'], device)
                    loss += l
                    avg_prob.append(probs)
                loss = loss/nr_words
                seq_loss += loss
                beginning_char = end_char 
                end_char = end_char + config['word_length']
            avg_seq_loss.append((seq_loss*data_word.shape[0])/data_word.shape[1])

            hidden_state = repackage_hidden(hidden_state)
            hidden_char = repackage_hidden(hidden_char)
            hidden_generator = repackage_hidden(hidden_generator)
    
    return np.mean(avg_seq_loss), np.mean(avg_prob)


def check_chars(input, target):
    """
    helper method to test input and target, not used 
    """
    list_input = []
    list_target = []
    for i, x in enumerate(input.T):
        for id in x:
            list_input.append(corpus.dictionary.idx2char[int(id)])
    for i,x in enumerate(target.T):
        for id in x:
            list_target.append(corpus.dictionary.idx2char[int(id)])
    print(list_input)
    print(list_target)

def generate_word(hidden_state, hidden_generator, target, last_idx, word_l, device = device):
    """
    old method to generate one word at a time instead of batch size * word 
    not used (too slow?)
    """
    last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device)
    probs_of_word = []
    word_loss = 0
    
    word_str = ''
    target_str = ''

    target = target[target.nonzero()].squeeze()
    if target.shape[0] > word_l:
        max_l = word_l
    else:
        max_l = target.shape[0]
  
    for i in range(max_l):
        out, hidden_generator = generator(last_char, hidden_state, hidden_generator)
        t = target[i].unsqueeze(0)
        target_str = target_str + corpus.dictionary.idx2char[t]
        out = out.view(1,-1)
        if torch.cuda.is_available():
            t = t.cuda()
        l = criterion(out, t)
        word_loss += l
        last_char = softmax(out)
        probs_of_word.append(torch.max(last_char, dim=1).values.item())
        last_char = torch.argmax(last_char,dim=1).squeeze()
        word_str = word_str + corpus.dictionary.idx2char[last_char.item()]
        #if last_char.item() == last_idx:
         #   break
    if i == 0:
        i = 1
    target_str = ''.join([corpus.dictionary.idx2char[t] for t in target])
    return word_loss/i, probs_of_word, word_str

def train(data):
    """
    method to train model and generator
    data_w: train data, word level
    data_c: train data, char level  
    """
    model.train()

    hidden_state = model.init_hidden(config['bs'])
    hidden_char = model.charEncoder.init_hidden(config['bs'])
    hidden_generator = generator.init_hidden(1) # only one word at a time 
    for batch_ndx, sample in enumerate(data):
        # get batch for word and character data
        data_word = sample['words']
        data_char = sample['chars']
        print(data_word.shape, data_char.shape)
        model.zero_grad()

        beginning_char = 0
        end_char = config['word_length']
        seq_loss = 0
        # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
        for id in range(data_word.shape[1]-1): # -1 because after final word no word to predict 
            # get the correct characters for word 'id'
            word_current = data_word[:,id].long()
            chars_current = data_char[:,beginning_char:end_char].long()
            chars_target = data_char[:,end_char:(end_char+config['word_length'])].long()

            #print('SHAPESS', word_current.shape, chars_target.shape, chars_current.shape)
            #data_char_target_word = target_char[:,end_char:end_char+config['bs']] 
            if torch.cuda.is_available():
                word_current.cuda()
                chars_current.cuda()
            # send word, characters to main LSTM forward call
            output, hidden_state, hidden_char = model(word_current, chars_current, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
            

            # generate next word based on hidden_state of LSTM with generator 
            #lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
            loss = 0
            #nr_words = hidden_state[0].shape[1]-1
            nr_words = word_current.shape[0]
            #print(hidden_state[0].shape)
            for word_nr in range(nr_words-1): # 34 
                #wl = lengths[word_nr]
                char_input = chars_current[word_nr].unsqueeze(0)
                #print('char input', char_input.shape)
                hs = hidden_state[0][1, word_nr, :] #[:, word_nr] # 1 x hidden size
                #hs = hidden_state[0]
                t = chars_target[word_nr] # t is of size word_length, padded with 0s at the moment
                stringy = [corpus.dictionary.idx2char[ti] for ti in t]
                # words contain <eow> token at the end of word (final char)
                l, _,outcome= generate_word(hs, hidden_generator, t, eow,config['word_length'], device)
                loss += l
            loss = loss/nr_words
            seq_loss += loss
            beginning_char = end_char 
            end_char = end_char + config['word_length']
        seq_loss = (seq_loss*data_word.shape[0])/data_word.shape[1]
        logging.info('batch {}, loss {}'.format(batch_ndx, seq_loss))
        seq_loss.backward()
        optimizer.step()
        hidden_state = repackage_hidden(hidden_state)
        hidden_char = repackage_hidden(hidden_char)
        hidden_generator = repackage_hidden(hidden_generator)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('config_path')
    args = argp.parse_args()
    config = toml.load(args.config_path)
    print(config)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                    logging.FileHandler(config['log_path'])])
    logging.info(config)
    try:
        os.makedirs(config['model_dir'])
    except:
        print('dict already exists')
    model_path_lstm = os.path.join(config['model_dir'], 'lstmmain')
    model_path_generator = os.path.join(config['model_dir'], 'generator')
    logging.info('model paths: {}, {}'.format(model_path_generator, model_path_lstm))
    logging.info('start loading corpus')
    corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=2)

    data_loader_train = DataLoader(Data(corpus.train_words, corpus.train_chars, config['word_length'], config['seq_len']), batch_size=config['bs'])
    data_loader_valid = DataLoader(Data(corpus.valid_words, corpus.valid_chars, config['word_length'], config['seq_len']), batch_size=config['bs'])
    data_loader_test = DataLoader(Data(corpus.test_words, corpus.test_chars, config['word_length'], config['seq_len']), batch_size=config['bs'])
    # corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=2)
    print('finished with loading corpus')
    seq_len = config['seq_len']
    l_r = config['l_r']
    word_length = config['word_length']
    
    
    gpu = False
    if torch.cuda.is_available():
        gpu = True


    # load datasets as batches for both word level and char level 
    #train_w, train_c = batchify(corpus.train_words, corpus.train_chars, config['bs'], gpu)
    #test_w, test_c = batchify(corpus.test_words, corpus.test_chars, config['bs'], gpu)
    #val_w, val_c = batchify(corpus.valid_words, corpus.valid_chars,config['bs'], gpu)


    params_char = [len(corpus.dictionary.idx2char), config['charmodel']['embedding_size'],  config['charmodel']['hidden_size'], config['charmodel']['dropout'], config['charmodel']['nlayers'], 'LSTM'] #(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):

    # initialise models

    model = Encoder(config['wordmodel']['dropout'], config['wordmodel']['embedding_size'], config['wordmodel']['hidden_size'], config['wordmodel']['nlayers'], len(corpus.dictionary.idx2word),params_char )
    #generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'])
    generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'])

    if gpu:
        model.cuda()
        generator.cuda()
    softmax = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = l_r)
    lr = config['l_r']
    eval_batch_size = config['bs']
    epochs = config['nr_epochs']

    eow = int(corpus.dictionary.char2idx['<eow>'])
    #print(eow)
    try:
        best_val_loss = None
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(data_loader_train)
            print(model.decoder.weight.data)
            val_loss, probs = evaluate(data_loader_valid)
            logging.info('-' * 89)
            logging.info('after epoch {}: average word probability: {}, validation loss: {}, time: {:5.2f}s'.format(epoch, np.mean(probs), val_loss, time.time() - epoch_start_time))
            logging.info('-' * 89)
            #print('VAL LOSS', val_loss)
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_path_lstm, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = val_loss
                with open(model_path_generator, 'wb') as f:
                    torch.save(generator, f)
            else:
                logging.info('after epoch {}, no improvement lr {} will be reduced'.format(epoch, str(lr)))
                lr /= 4.0
    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')
        # Load the best saved model.
        with open(model_path_lstm, 'rb') as f:
            model = torch.load(f)
        with open(model_path_generator, 'rb') as f:
            generator = torch.load(f)
    # Run on test data.
    test_loss, probs = evaluate(data_loader_test)
    logging.info('*' * 89)
    logging.info('End of training: average word probability: {}, validation loss: {}'.format(np.mean(probs), val_loss))
    logging.info('*' * 89)
