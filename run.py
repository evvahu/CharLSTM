import argparse
from dict_old import Corpus, Dictionary
from utilsy import batchify, get_batch, get_char_input, repackage_hidden
from models_new import Encoder, CharGenerator
import torch
import numpy as np
import multiprocessing as mp
import logging
import time 
import toml 
import os 

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
corpus = Corpus(config['path'])
print('finished with loading corpus')

seq_len = config['seq_len']
l_r = config['l_r']
word_length = config['word_length']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
gpu = False
if torch.cuda.is_available():
    gpu = True
#cpu_count = 6
#if mp.cpu_count() < cpu_count:
#    cpu_count = mp.cpu_count()
train_d = batchify(corpus.train, config['bs'], gpu)
test_d = batchify(corpus.test, config['bs'], gpu)
val_d = batchify(corpus.valid,config['bs'], gpu)
print(train_d.size())
params_char = [len(corpus.dictionary.idx2char), config['charmodel']['embedding_size'],  config['charmodel']['hidden_size'], config['charmodel']['dropout'], config['charmodel']['nlayers'], 'LSTM'] #(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
print('corpus dict', corpus.dictionary.idx2char[0])
print(corpus.dictionary.char2idx)
print(corpus.dictionary.idx2char)
model = Encoder(config['wordmodel']['dropout'], config['wordmodel']['embedding_size'], config['wordmodel']['hidden_size'], config['wordmodel']['nlayers'], len(corpus.dictionary.idx2word),device, params_char )
generator = CharGenerator(config['wordmodel']['hidden_size'], config['generator']['embedding_size'],len(corpus.dictionary.idx2char),  config['generator']['hidden_size'], config['generator']['nlayers'], config['generator']['dropout'], device)
#generator = CharGenerator(100, 100,, 100, 1, 0.1, device) # hl_size, emb_size, nchar, nhid, nlayers, dropout, rnn_type = 'LSTM'
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
print(eow)


def evaluate(data_source):
    model.eval()
    hidden_state = model.init_hidden(eval_batch_size)
    hidden_char = model.charEncoder.init_hidden(eval_batch_size) # bs 
    hidden_generator = generator.init_hidden(1)
    eval_loss = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data_word, targets = get_batch(data_source, i, seq_len)
            data_char = get_char_input(data_word, corpus.dictionary,device, word_length) 
            hidden_state = repackage_hidden(hidden_state)
            id_char = 0
            beginning_char = 0 
            end_char = config['bs']
            all_ws = 0
            seq_loss = 0
            for id in range(data_word.shape[0]-1):
                w_loss = 0
                data_char_part = data_char[:,beginning_char:end_char]
                data_char_target_word = data_char[:,end_char:end_char+config['bs']]
                hidden_state = repackage_hidden(hidden_state)
                hidden_char = repackage_hidden(hidden_char)
                lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
                #print(data_char_part)
                #last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device).repeat(bs)
                # send in each word individually so that we can retrieve hidden state at each word 
                out, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char) 
                for word_nr in range(0, hidden_state[0].shape[1]):
                    word_l = lengths[word_nr]
                    b = hidden_state[0][:, word_nr]
                    t = data_char_target_word[:, word_nr]
                    l, probs, _ = generate_word(b, hidden_generator, t, eow, word_l, device=device) #change word leng
                    w_loss += l
                #id_char = id_char + bs
                seq_loss += w_loss * config['bs']
                beginning_char = end_char
                end_char = beginning_char + config['bs']
            eval_loss += seq_loss/seq_len
            
           
    return eval_loss/data_source.size(0), probs


def generate_word(hidden_state, hidden_generator, target, last_idx, word_l, device = device):
    last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device)
    probs_of_word = []
    word_loss = 0
    if word_l > len(target):
        word_l = len(target)
    word_str = ''
    target_str = ''
    for i in range(word_l):
        out, hidden_generator = generator(last_char, hidden_state, hidden_generator)
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

def train(data_source):
    model.train()
    #p = mp.Pool(mp.cpu_count())
    for batch, i in enumerate(range(0, train_d.size(0) - 1, seq_len)):
        print('batch', batch)
        data_word, targets = get_batch(data_source, i, seq_len) # data word : sentence length x batch size 
        data_char = get_char_input(data_word, corpus.dictionary,device,eow, word_length) # data char (each word in one column): max word length x (seq_len*batchsize) 
        model.zero_grad()
        #initialise hidden states, hidden_state tuple of hidden state and cell sttate 
        hidden_state = model.init_hidden(config['bs'])
        hidden_char = model.charEncoder.init_hidden(config['bs'])
        hidden_generator = generator.init_hidden(1) # only one word at a time 
        beginning_char = 0 
        end_char = config['bs']
        seq_loss = 0
        # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
        for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
            data_char_part = data_char[:,beginning_char:end_char] # get batch size big blocks of chars (for word at position *id*)
            data_char_target_word = data_char[:,end_char:end_char+config['bs']] # get batch size big blocks of chars (for word at position *id + 1*)
            #print('data char shape: for input and target', data_char_part.shape, data_char_target_word.shape)
            #print('word shape: it should be bs * 1', data_word[id].shape)
            output, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
            # hidden_state size: 1,6,100 batch size * hidden size 
            lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
            loss = 0

            # GENERATE NEXT WORD: loop over all words in batch (could already be done one step further up) and generate word at time point n+1 
            # one could parallelise the generation process (problem: don't want to detach it from current graph)
            for word_nr in range(hidden_state[0].shape[1]):
                wl = lengths[word_nr]
                hs = hidden_state[0][:, word_nr] # 1 x hidden size 
                t = data_char_target_word[:, word_nr] # t is of size word_length, padded with 0s at the moment
                stringy = [corpus.dictionary.idx2char[ti] for ti in t]
                # words contain <eow> token at the end of word (final char)
                l, _,_ = generate_word(hs, hidden_generator, t, eow,wl, device=device)
                loss += l


            # @TODO: one could include a word level predictor here 
            beginning_char = end_char
            end_char = beginning_char + config['bs']
            seq_loss += loss
        seq_loss = seq_loss/(seq_len*config['bs'])
        print('batch loss', seq_loss)
        #print(seq_loss)
        #print('loss after batch {}: {}'.format(batch, seq_loss))
        seq_loss.backward()
        optimizer.step()
        hidden_state = repackage_hidden(hidden_state)
        hidden_char = repackage_hidden(hidden_char)
        hidden_generator = repackage_hidden(hidden_generator)


try:
    best_val_loss = None
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(train_d)
        val_loss, probs = evaluate(val_d)
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
test_loss, probs = evaluate(test_d)
logging.info('*' * 89)
logging.info('End of training: average word probability: {}, validation loss: {}'.format(np.mean(probs), val_loss))
logging.info('*' * 89)
