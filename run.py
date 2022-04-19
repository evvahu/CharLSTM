
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
import math
from utilsy import MyDataParallel

def evaluate(data):
    pass


def evaluate(data, model):
    model.eval()
    #losses = []
    total_loss = 0
    with torch.no_grad():
        hidden_state = model.init_hidden(config['bs'])
        hidden_generator = model.decoder.init_hidden(config['word_length']) # only one word at a time 
        for batch_ndx, sample in enumerate(data):
            if torch.cuda.is_available():
                current_device = 'cuda:{}'.format(torch.cuda.current_device())
            else:
                current_device = 'cpu'
            model = model.to(current_device)
            # get batch for word and character data
            data_word = sample['words']
            if data_word.shape[0] < config['bs']:
                continue
            
            data_char = sample['chars'][:,config['word_length']:] # has to be next word
            data_char = data_char.reshape(config['bs'], (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
            data_target = sample['target'][:, config['word_length']:]
            data_target = data_target.reshape(config['bs'], (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
            model.zero_grad()
            output, hidden_state, hidden_generator = model(data_word, data_char, hidden_state, hidden_generator, current_device)
            loss = 0
            for i, pred in enumerate(output):
                t = data_target[:,i,:].long()
                if torch.cuda.is_available():
                    t = t.cuda()
                
                #print('p shape t shape', pred.shape, t.shape) # pred have to be of shape bs x nclasses x seq_len
                loss+= torch.nn.CrossEntropyLoss()(pred, t)
            total_loss += loss
            #loss = loss/len(output)
            #losses.append(loss.item())
            hidden_state = repackage_hidden(hidden_state)
            hidden_generator = repackage_hidden(hidden_generator)
    return total_loss/ (len(data) -1)#np.mean(losses)


def train(data, model):
    """
    method to train model and generator
    data_w: train data, word level
    data_c: train data, char level  
    """
    model.train()

    hidden_state = model.init_hidden(config['bs'])
    hidden_generator = model.decoder.init_hidden(config['word_length']) # only one word at a time
    outputs = [] 
    for batch_ndx, sample in enumerate(data):
        # get batch for word and character data
        if torch.cuda.is_available():
            current_device = 'cuda:{}'.format(torch.cuda.current_device())
        else:
            current_device = 'cpu'
        data_word = sample['words']
        if data_word.shape[0] < config['bs']:
            continue
        #model = model.to(current_device)
        data_char = sample['chars'][:,config['word_length']:] # has to be next word
        data_char = data_char.reshape(-1, (config['seq_len']-1), config['word_length']) 
        data_target = sample['target'][:, config['word_length']:]
        data_target = data_target.reshape(-1, (config['seq_len']-1), config['word_length']) 
        model.zero_grad()
        print('batch idx: {}, data shape: {}, char shape: {}'.format(batch_ndx, data_word.shape, data_char.shape))
        output, hidden_state, hidden_generator = model(data_word, data_char, hidden_state, hidden_generator, current_device)
        loss = 0
        for i, pred in enumerate(output):
            t = data_target[:,i,:].long()
            if torch.cuda.is_available():
                t = t.cuda()
            #print('p shape t shape', pred.shape, t.shape) # pred have to be of shape bs x nclasses x seq_len
            loss+= criterion(pred, t)
        loss = loss/len(output)
        print('batch nr: {}, loss:{}'.format(batch_ndx, loss))
        loss.backward()
        optimizer.step()
        hidden_state = repackage_hidden(hidden_state)
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
        print('model directory already exists')
    model_path_lstm = os.path.join(config['model_dir'], 'lstmmain')
    model_path_generator = os.path.join(config['model_dir'], 'generator')
    logging.info('model paths: {}, {}'.format(model_path_generator, model_path_lstm))
    logging.info('start loading corpus')
    logging.info('cpu count:{}'.format(mp.cpu_count()))
    print('cpu count', mp.cpu_count)
    #corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=mp.cpu_count())
    corpus = Corpus(config['path'], config['word_length'], config['seq_len'], parallel=False)
    logging.info('finished loading corpus')
    data_loader_train = DataLoader(Data(corpus.train_words, corpus.train_chars, corpus.train_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
    data_loader_valid = DataLoader(Data(corpus.valid_words, corpus.valid_chars,corpus.valid_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
    data_loader_test = DataLoader(Data(corpus.test_words, corpus.test_chars, corpus.test_targets, config['word_length'], config['seq_len']), batch_size=config['bs'])
    # corpus = Corpus(config['path'], config['word_length'], config['seq_len'], cpu_count=2)
    logging.info('finished with data loader')
    seq_len = config['seq_len']
    l_r = config['l_r']
    word_length = config['word_length']
    
    
    gpu = False
    if torch.cuda.is_available():
        gpu = True
   # params_char = [len(corpus.dictionary.idx2char), config['charmodel']['embedding_size'],  config['charmodel']['hidden_size'], config['charmodel']['dropout'], config['charmodel']['nlayers'], 'LSTM'] #(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
    params_char = [config['wordmodel']['hidden_size'], config['charmodel']['embedding_size'], len(corpus.dictionary.idx2char), config['charmodel']['hidden_size'],config['charmodel']['nlayers'], config['charmodel']['dropout'], 0, 'LSTM']  
    #self, hl_size, emb_size, nchar, nhid, nlayers, dropout,padding_id=0, rnn_type = 'LSTM'): 
    
    # initialise models

    model = Encoder(config['wordmodel']['dropout'], config['wordmodel']['embedding_size'], config['wordmodel']['hidden_size'], config['wordmodel']['nlayers'], len(corpus.dictionary.idx2word),params_char )
    #generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'])
    #generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'])

    if torch.cuda.device_count() > 1:
        model = MyDataParallel(model)
        #model = model.cuda()
    softmax = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = l_r)
    lr = config['l_r']
    eval_batch_size = config['bs']
    epochs = config['nr_epochs']

    eow = int(corpus.dictionary.char2idx['<eow>'])
    #print(eow)
    logging.info('start train')


    try:
        best_val_loss = None
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(data_loader_train, model)
            val_loss = evaluate(data_loader_valid, model)
            logging.info('-' * 89)
            logging.info('after epoch {}: validation loss: {}, perplexity: {},  time: {:5.2f}s'.format(epoch, val_loss, math.exp(val_loss), time.time() - epoch_start_time))
            logging.info('-' * 89)
            #print('VAL LOSS', val_loss)
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_path_lstm, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = val_loss
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
    test_loss = evaluate(data_loader_test, model)
    logging.info('*' * 89)
    logging.info('End of training: average word probability: {}, test loss: {}'.format(0, test_loss))
    logging.info('*' * 89)
