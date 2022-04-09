import argparse
from data import Corpus, Dictionary
from utilsy import batchify, get_batch, get_char_input, repackage_hidden, get_char_batch
from models import Encoder, CharGenerator
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
corpus = Corpus(config['path'], config['word_length'])
print('finished with loading corpus')

seq_len = config['seq_len']
l_r = config['l_r']
word_length = config['word_length']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
gpu = False
if torch.cuda.is_available():
    gpu = True


# load datasets as batches for both word level and char level 
train_w, train_c = batchify(corpus.train_words, corpus.train_chars, config['bs'], gpu)
test_w, test_c = batchify(corpus.test_words, corpus.test_chars, config['bs'], gpu)
val_w, val_c = batchify(corpus.valid_words, corpus.valid_chars,config['bs'], gpu)


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


def evaluate(data_w, data_c):
    """
    see train() for analogous code 
    """
    model.eval()
    #p = mp.Pool(mp.cpu_count())
    end_char_i = 0
    avg_prob = []
    avg_seq_loss = []
    hidden_state = model.init_hidden(config['bs'])
    hidden_char = model.charEncoder.init_hidden(config['bs'])
    hidden_generator = generator.init_hidden(1) # only one word at a time 
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_w.size(0) - 1, seq_len)):
            # get batch for word and character data
            data_word, target_word = get_batch(data_w, i, seq_len) # data word : sentence length x batch size
            data_char, target_char, end_char_i = get_char_batch(data_c, end_char_i, seq_len, config['word_length'])
            beginning_char = 0 
            #end_char = config['bs']
            end_char = config['word_length']
            seq_loss = 0
            # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
            for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
                # get the correct characters for word 'id'
                data_char_part = data_char[beginning_char:end_char,] # one word per column, column = batch size 
                data_word_part = data_word[id]
                data_char_target_word = data_char[end_char:end_char+config['bs'],:] 
                #data_char_target_word = target_char[:,end_char:end_char+config['bs']] 
                if torch.cuda.is_available():
                    data_char_part.cuda()
                    data_word_part.cuda()
                # send word, characters to main LSTM forward call
                output, hidden_state, hidden_char = model(data_word_part, data_char_part, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
                # generate next word based on hidden_state of LSTM with generator 
                lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
                loss = 0
                for word_nr in range(hidden_state[0].shape[1]): # 34 
                    wl = lengths[word_nr]
                    hs = hidden_state[0][:, word_nr] # 1 x hidden size
                    t = data_char_target_word[:, word_nr] # t is of size word_length, padded with 0s at the moment
                    stringy = [corpus.dictionary.idx2char[ti] for ti in t]
                    # words contain <eow> token at the end of word (final char)
                    l, prob,outcome= generate_word(hs, hidden_generator, t, eow,wl, device)
                    avg_prob.append(np.mean(prob))
                    loss += l
                loss = loss/word_nr 
                seq_loss += loss
                beginning_char = end_char
                end_char = beginning_char + config['word_length']

            seq_loss = (seq_loss*data_word.shape[0])/data_word.shape[1]
            avg_seq_loss.append(seq_loss.item())
            print('batch {}, loss {}'.format(batch, seq_loss))
            hidden_state = repackage_hidden(hidden_state)
            hidden_char = repackage_hidden(hidden_char)
            hidden_generator = repackage_hidden(hidden_generator)
    
    return np.mean(avg_seq_loss), np.mean(avg_prob)

def generate_word_bs(hidden_state, input, hidden_generator, device=device):
    """
    hidden_state: hidden state of main LSTM model
    input: character input, batch_size X maximum length of word (each cell contains the index of a character, padded with 0s)
    hidden_generator: initialised hidden_state for generator lstm
    device: current device 
    """
    # input, coudl look something like [2,3,4,5,EOW, 0,0,0,0,0] 
    # target [3,4,5,EOW, 0,0,0, additional_0]
    
    out, hidden_generator = generator(input, hidden_state, hidden_generator)
    target = input[1:,:] # from input first item deleted and one row of zeros added
    #print(input.shape, target.shape)
    #check_chars(input,target)
    target= torch.cat((target.to(dtype=int).to(device), torch.zeros(1, input.shape[1], dtype=int, device=device))).T
    #print(target.shape, out.shape)
    probs = torch.max(softmax(out), dim=2)[0] # seq_len * bs * nr_classes probs_of_word.append(torch.max(last_char, dim=1).values.item()) 
    out = out.reshape(out.shape[1], out.shape[2], -1)
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

def generate_word(hidden_state, hidden_generator, target, last_idx, word_l, device = device):
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
        if last_char.item() == last_idx:
            break
    if i == 0:
        i = 1
    target_str = ''.join([corpus.dictionary.idx2char[t] for t in target])
    print(target_str, word_str)
    #print(word_str, target_str)
    return word_loss/i, probs_of_word, word_str

def train(data_w, data_c):
    """
    method to train model and generator
    data_w: train data, word level
    data_c: train data, char level  
    """
    model.train()
    #p = mp.Pool(mp.cpu_count())
    end_char_i = 0
    hidden_state = model.init_hidden(config['bs'])
    hidden_char = model.charEncoder.init_hidden(config['bs'])
    hidden_generator = generator.init_hidden(1) # only one word at a time 
    for batch, i in enumerate(range(0, data_w.size(0) - 1, seq_len)):
        # get batch for word and character data
        data_word, target_word = get_batch(data_w, i, seq_len) # data word : sentence length x batch size
        data_char, target_char, end_char_i = get_char_batch(data_c, end_char_i, seq_len, config['word_length'])
<<<<<<< HEAD
        #data_char = get_char_input(data_word, corpus.dictionary,device,eow, word_length) # data char (each word in one column): max word length x (seq_len*batchsize) 
=======
>>>>>>> word_individual
        model.zero_grad()
        beginning_char = 0 
        end_char = config['word_length']
        seq_loss = 0
        # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
        for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
            # get the correct characters for word 'id'
            data_char_part = data_char[beginning_char:end_char,] # one word per column, column = batch size
            print(data_char_part.shape) 
            data_word_part = data_word[id]
            data_char_target_word = data_char[end_char:end_char+config['bs'],:] 
            #data_char_target_word = target_char[:,end_char:end_char+config['bs']] 
            if torch.cuda.is_available():
                data_char_part.cuda()
                data_word_part.cuda()
            # send word, characters to main LSTM forward call
            output, hidden_state, hidden_char = model(data_word_part, data_char_part, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
            # generate next word based on hidden_state of LSTM with generator 
            lengths = [len(corpus.dictionary.idx2word[ix])+1 for ix in data_word[id+1]]
            loss = 0
            for word_nr in range(hidden_state[0].shape[1]): # 34 
                wl = lengths[word_nr]
                hs = hidden_state[0][:, word_nr] # 1 x hidden size
                t = data_char_target_word[:, word_nr] # t is of size word_length, padded with 0s at the moment
                stringy = [corpus.dictionary.idx2char[ti] for ti in t]
                # words contain <eow> token at the end of word (final char)
                l, _,outcome= generate_word(hs, hidden_generator, t, eow,wl, device)
                loss += l
            loss = loss/word_nr 
            seq_loss += loss
            beginning_char = end_char
            end_char = beginning_char + config['word_length']
        seq_loss = (seq_loss*data_word.shape[0])/data_word.shape[1]
        logging.info('batch {}, loss {}'.format(batch, seq_loss))
        seq_loss.backward()
        optimizer.step()
        hidden_state = repackage_hidden(hidden_state)
        hidden_char = repackage_hidden(hidden_char)
        hidden_generator = repackage_hidden(hidden_generator)


try:
    best_val_loss = None
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(train_w, train_c)
        val_loss, probs = evaluate(val_w, val_c)
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
test_loss, probs = evaluate(test_w, test_c)
logging.info('*' * 89)
logging.info('End of training: average word probability: {}, validation loss: {}'.format(np.mean(probs), val_loss))
logging.info('*' * 89)
