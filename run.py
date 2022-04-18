
import argparse
from data import Corpus, Dictionary
<<<<<<< HEAD
from utilsy import batchify, get_batch, get_char_input, repackage_hidden, get_char_batch, MyDataParallel
=======
from data_loader import Data
from utilsy import batchify, get_batch, get_char_input, repackage_hidden, get_char_batch
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
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

<<<<<<< HEAD
argp = argparse.ArgumentParser()

# load config with hyperparameters, paths etc..
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
corpus = Corpus(config['path'], config['word_length'])
print('finished with loading corpus')

seq_len = config['seq_len']
l_r = config['l_r']
word_length = config['word_length']
gpu = False
if torch.cuda.is_available():
    gpu = True

# load datasets as batches for both word level and char level 
train_w, train_c = batchify(corpus.train_words, corpus.train_chars, config['bs'])#, gpu)
test_w, test_c = batchify(corpus.test_words, corpus.test_chars, config['bs'])#, gpu)
val_w, val_c = batchify(corpus.valid_words, corpus.valid_chars,config['bs'])#, gpu)
print(train_w.size())
params_char = [len(corpus.dictionary.idx2char), config['charmodel']['embedding_size'],  config['charmodel']['hidden_size'], config['charmodel']['dropout'], config['charmodel']['nlayers'], 'LSTM'] #(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
print('corpus dict', corpus.dictionary.idx2char[0])
print(corpus.dictionary.char2idx)
print(corpus.dictionary.idx2char)

# initialise models 
model = Encoder(config['wordmodel']['dropout'], config['wordmodel']['embedding_size'], config['wordmodel']['hidden_size'], config['wordmodel']['nlayers'], len(corpus.dictionary.idx2word),params_char)
generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'])

# for parallel processing, if more than one device is available: 
if torch.cuda.device_count() > 1:
    model = MyDataParallel(model)
    generator = MyDataParallel(generator)
logging.info('DEVICE COUNT: {}'.format(torch.cuda.device_count()))

softmax = torch.nn.Softmax(dim=1)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = l_r)
lr = config['l_r']
eval_batch_size = config['bs']
epochs = config['nr_epochs']

eow = int(corpus.dictionary.char2idx['<eow>'])
print(eow)


def evaluate(data_w, data_c):
    """
    see train() for analogous code 
    """
    avg_prob = []
    avg_seq_loss = []
    end_char_i = 0
    model.eval()
    generator.eval()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device = 'cuda:{}'.format(device)
    else:
        device = 'cpu'
=======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(data):
    pass


def evaluate(data):
    model.eval()
    #losses = []
    total_loss = 0
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
    with torch.no_grad():
        hidden_state = model.init_hidden(config['bs'])
        hidden_generator = model.decoder.init_hidden(config['word_length']) # only one word at a time 
        for batch_ndx, sample in enumerate(data):
            # get batch for word and character data
            data_word = sample['words']
            if data_word.shape[0] < config['bs']:
                continue
            data_char = sample['chars'][:,config['word_length']:] # has to be next word
            data_char = data_char.reshape(config['bs'], (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
            data_target = sample['target'][:, config['word_length']:]
            data_target = data_target.reshape(config['bs'], (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
            model.zero_grad()
<<<<<<< HEAD
            #initialise hidden states, hidden_state tuple of hidden state and cell sttate 
            hidden_state = model.init_hidden(config['bs'])
            hidden_char = model.charEncoder.init_hidden(config['bs'])
            hidden_generator = generator.init_hidden(config['bs']) # only one word at a time 
            beginning_char = 0 
            #end_char = config['bs']
            end_char = config['word_length']
            seq_loss = 0
            # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
            for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
                #data_char_part = data_char[:,beginning_char:end_char] # get batch size big blocks of chars (for word at position *id*)
                data_char_part = data_char[beginning_char:end_char,] # one word per column, column = batch size 
                #print('data char shape: for input and target', data_char_part.shape, data_char_target_word.shape)
                #print('word shape: it should be bs * 1', data_word[id].shape)
                output, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char, device) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
                # hidden_state size: 1,6,100 batch size * hidden size
                word_loss, probs  = generate_word_bs(hidden_state, data_char_part, hidden_generator, device)
                avg_prob.append(torch.mean(probs).item())
                seq_loss += word_loss
                beginning_char = end_char
                end_char = beginning_char + config['word_length']
            seq_loss = (seq_loss*data_word.shape[0])/data_word.shape[1]
            avg_seq_loss.append(seq_loss.item())
            
=======
            output, hidden_state, hidden_generator = model(data_word, data_char, hidden_state, hidden_generator)
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
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
            hidden_state = repackage_hidden(hidden_state)
            hidden_generator = repackage_hidden(hidden_generator)
<<<<<<< HEAD
        
    return np.mean(avg_seq_loss), np.mean(avg_prob)

def generate_word_bs(hidden_state, input, hidden_generator, device):
    """
    hidden_state: hidden state of main LSTM model
    input: character input, batch_size X maximum length of word (each cell contains the index of a character, padded with 0s)
    hidden_generator: initialised hidden_state for generator lstm
    device: current device 
    """
    # input, coudl look something like [2,3,4,5,EOW, 0,0,0,0,0] 
    # target [3,4,5,EOW, 0,0,0, additional_0]
    
    out, hidden_generator = generator(input, hidden_state, hidden_generator,device)
    target = input[1:,:].to(device) # from input first item deleted and one row of zeros added
    #print(input.shape, target.shape)
    #check_chars(input,target)
    target= torch.cat((target.to(dtype=int), torch.zeros(1, input.shape[1], dtype=int, device=device))).T
    #print(target.shape, out.shape)
    probs = torch.max(softmax(out), dim=2)[0] # seq_len * bs * nr_classes 
    out = out.reshape(out.shape[1], out.shape[2], -1)
    # out size: batch_size X nr_of_classes X max length of word, target: batch_size X max length of word 
    word_loss = criterion(out, target)
    return word_loss, probs

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

def generate_word(hidden_state, hidden_generator, target, last_idx, word_l, device):
    """
    old method to generate one word at a time instead of batch size * word 
    not used (too slow?)
    """
    last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device)
    probs_of_word = []
    word_loss = 0
    if word_l > len(target):
        word_l = len(target)
    word_str = ''
    target_str = ''
    for i in range(word_l):
        out, hidden_generator = generator(last_char, hidden_state, hidden_generator, device)
        #target = target[i].unsqueeze(0)
        t = target[i].unsqueeze(0)
        target_str += corpus.dictionary.idx2char[t]
        out = out.view(1,-1)
        l = criterion(out, t)
        word_loss += l
        last_char = softmax(out)
        probs_of_word.append(torch.max(last_char, dim=1).values.item())
        #last_char = torch.squeeze(torch.argmax(last_char, dim=1))
        last_char = torch.argmax(last_char,dim=1).squeeze()
        word_str += corpus.dictionary.idx2char[last_char.item()]
        if last_char.item() == last_idx:
            #print('broken after eow', last_char, last_idx, i)
            break
    if i == 0:
        i = 1
    #print(word_str, target_str)
    return word_loss/i, probs_of_word, word_str

def train(data_w, data_c):
=======
    return total_loss/ (len(data) -1)#np.mean(losses)


def train(data):
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
    """
    method to train model and generator
    data_w: train data, word level
    data_c: train data, char level  
    """
<<<<<<< HEAD
    
    end_char_i = 0
    for batch, i in enumerate(range(0, data_w.size(0) - 1, seq_len)):
        # select the correct device 
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device = 'cuda:{}'.format(device)
        else:
            device = 'cpu'
        logging.info('CURRENT DEVICE: {}'.format(device))
        # get batch for word and character data
        data_word, _= get_batch(data_w, i, seq_len) # data word : seq length x batch size
        data_char, _, end_char_i = get_char_batch(data_c, end_char_i, seq_len, config['word_length']) # data char: (seq_length*max_word_length) X batch_size

        # initialise hidden states, hidden_state tuple of hidden state and cell state
        # send everythign to curretn device
        model.to(device)
        generator.to(device)
        model.train()
        generator.train()
        generator.zero_grad()
        model.zero_grad()
        generator.to(device)
        model.charEncoder.to(device) 
        hidden_state = [h.to(device) for h in model.init_hidden(config['bs'])]
        hidden_char = [h.to(device) for h in model.charEncoder.init_hidden(config['bs'])]
        hidden_generator = [h.to(device) for h in generator.init_hidden(config['bs'])] # only one word at a time 
        beginning_char = 0 
        end_char = config['word_length']
        seq_loss = 0

        # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
        for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
            # get the correct characters for word 'id'
            data_char_part = data_char[beginning_char:end_char,] # max length of word X batch_size (always word at id point in sequence)
            # send word, characters to main LSTM forward call
            print('data chaaar', data_char_part.shape)
            output, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char, device) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
            # generate next word based on hidden_state of LSTM with generator 
            word_loss, probs  = generate_word_bs(hidden_state, data_char_part, hidden_generator, device)
            seq_loss += word_loss
            beginning_char = end_char
            end_char = beginning_char + config['word_length']
        seq_loss = (seq_loss*data_word.shape[0])/data_word.shape[1]
        print('batch {}, loss {}'.format(batch, seq_loss))
        seq_loss.backward()
=======
    model.train()

    hidden_state = model.init_hidden(config['bs'])
    hidden_generator = model.decoder.init_hidden(config['word_length']) # only one word at a time 
    for batch_ndx, sample in enumerate(data):
        # get batch for word and character data
        data_word = sample['words']
        if data_word.shape[0] < config['bs']:
            continue
        data_char = sample['chars'][:,config['word_length']:] # has to be next word
        data_char = data_char.reshape(-1, (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
        data_target = sample['target'][:, config['word_length']:]
        data_target = data_target.reshape(-1, (config['seq_len']-1), config['word_length']) # @TODO check!!!!!!
        model.zero_grad()
        output, hidden_state, hidden_generator = model(data_word, data_char, hidden_state, hidden_generator)
        loss = 0
        for i, pred in enumerate(output):
            t = data_target[:,i,:].long()
            if torch.cuda.is_available():
                t = t.cuda()
            #print('p shape t shape', pred.shape, t.shape) # pred have to be of shape bs x nclasses x seq_len
            loss+= criterion(pred, t)
        loss = loss/len(output)
       # print('batch nr: {}, loss:{}'.format(batch_ndx, loss))
        loss.backward()
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
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

    if gpu:
        model.cuda()

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
            train(data_loader_train)
            val_loss = evaluate(data_loader_valid)
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
    test_loss = evaluate(data_loader_test)
    logging.info('*' * 89)
    logging.info('End of training: average word probability: {}, test loss: {}'.format(0, test_loss))
    logging.info('*' * 89)
