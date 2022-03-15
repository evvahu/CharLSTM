from dict_old import Corpus, Dictionary
from utilsy import batchify, get_batch, get_char_input, repackage_hidden, encode_sentence
from models_new import Encoder, CharGenerator
import torch
import numpy as np
import multiprocessing as mp
#self, dropout, ninp, nhid, nlayers, ntoken, params_char = '', rnn_type = 'LSTM'):
path = 'CharLSTMLM/testfiles'
#path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/wiki_splits_shortened'
model_path = 'CharLSTMLM/models/modelx'
corpus = Corpus(path)

bs = 6
seq_len = 4
l_r = 0.1
word_length = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
gpu = False
if torch.cuda.is_available():
    gpu = True
#cpu_count = 6
#if mp.cpu_count() < cpu_count:
#    cpu_count = mp.cpu_count()
train_d = batchify(corpus.train, bs, gpu)
test_d = batchify(corpus.test, bs, gpu)
val_d = batchify(corpus.valid,bs, gpu)
print(train_d.size())
params_char = [len(corpus.dictionary.idx2char), 100, 100, 0.1, 1, 'LSTM'] #(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
print('corpus dict', corpus.dictionary.idx2char[0])
print(len(corpus.dictionary.char2idx))
model = Encoder(0.1, 100, 100, 1, len(corpus.dictionary.idx2word), device, params_char) #(self, dropout, ninp, nhid, nlayers, ntoken, rnn_type = 'LSTM', models='CW'):
generator = CharGenerator(100, 100, len(corpus.dictionary.idx2char), 100, 1, 0.1, device) # hl_size, emb_size, nchar, nhid, nlayers, dropout, rnn_type = 'LSTM'
if gpu:
    model.cuda()
    generator.cuda()
softmax = torch.nn.Softmax(dim=1)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = l_r)
lr = 0.01
eval_batch_size = bs
epochs = 10

eow = corpus.dictionary.char2idx['<eow>']
print('eow', eow) 
def evaluate_sentences(sentence, model, generator):
    sent = encode_sentence()
    #for word in sentence:
    pass


def evaluate(data_source):
    model.eval()
    loss = 0
    hidden_state = model.init_hidden(eval_batch_size)
    hidden_char = model.charEncoder.init_hidden(eval_batch_size) # bs 
    hidden_generator = generator.init_hidden(1)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_len):
            data_word, targets = get_batch(data_source, i, seq_len)
            data_char = get_char_input(data_word, corpus.dictionary.idx2word, corpus.dictionary.char2idx,device, word_length) 
            hidden_state = repackage_hidden(hidden_state)
            id_char = 0
            beginning_char = 0 
            end_char = bs
            for id in range(data_word.shape[0]-1):
                data_char_part = data_char[:,beginning_char:end_char]
                data_char_target_word = data_char[:,end_char:end_char+bs]
                hidden_state = repackage_hidden(hidden_state)
                hidden_char = repackage_hidden(hidden_char)
                #print(data_char_part)
                last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device).repeat(bs)
                # send in each word individually so that we can retrieve hidden state at each word 
                out, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char) 
                for word_nr in range(hidden_state[0].shape[1]):
                    b = hidden_state[0][:, word_nr]
                    t = data_char_target_word[:, word_nr]
                    l, probs = generate_word(b, hidden_generator, t, device=device)
                    loss += l
                #id_char = id_char + bs
                beginning_char = end_char
                end_char = beginning_char + bs
           
            
    return loss


def generate_word(hidden_state, hidden_generator, target, device = device):
    last_char = torch.tensor(corpus.dictionary.char2idx['<bow>'], device=device)
    prob_of_word = []
    loss = 0
    for i in range(word_length + 1):
        out, hidden_generator = generator(last_char, hidden_state, hidden_generator)
        #target = target[i].unsqueeze(0)
        t = target[i].unsqueeze(0)
        out = out.view(1,-1)
        l = criterion(out, t)
        loss += l
        prob_of_word.append(out)
        last_char = torch.argmax(softmax(out), dim=1)
        last_char = torch.squeeze(last_char)
    return loss, prob_of_word

def train(data_source):
    model.train()
    #p = mp.Pool(mp.cpu_count())
    for batch, i in enumerate(range(0, train_d.size(0) - 1, seq_len)):
        data_word, targets = get_batch(data_source, i, seq_len) # data word : sentence length x batch size 
        data_char = get_char_input(data_word, corpus.dictionary.idx2word, corpus.dictionary.char2idx,device, word_length) # data char (each word in one column): max word length x (seq_len*batchsize) 
        model.zero_grad()
        #initialise hidden states, hidden_state tuple of hidden state and cell sttate 
        hidden_state = model.init_hidden(bs)
        hidden_char = model.charEncoder.init_hidden(bs)
        hidden_generator = generator.init_hidden(1) # only one word at a time 
        model.zero_grad()
        beginning_char = 0 
        end_char = bs

        # loop over every word: give each word individually to LSTM main model so that we can retrieve hidden state at each word 
        for id in range(data_word.shape[0]-1): # -1 because after final word no word to predict 
            data_char_part = data_char[:,beginning_char:end_char] # get batch size big blocks of chars (for word at position *id*)
            data_char_target_word = data_char[:,end_char:end_char+bs] # get batch size big blocks of chars (for word at position *id + 1*)
            #print('data char shape: for input and target', data_char_part.shape, data_char_target_word.shape)
            #print('word shape: it should be bs * 1', data_word[id].shape)
            output, hidden_state, hidden_char = model(data_word[id], data_char_part, hidden_state, hidden_char) #out: sequence length, batch size, out_size,  hi[0] contains final hidden state for each element in batch 
            # hidden_state size: 1,6,100 batch size * hidden size 

            loss = 0

            # GENERATE NEXT WORD: loop over all words in batch (could already be done one step further up) and generate word at time point n+1 
            for word_nr in range(hidden_state[0].shape[1]):
                hs = hidden_state[0][:, word_nr] # 1 x hidden size 
                t = data_char_target_word[:, word_nr] # t is of size word_length, padded with 0s at the moment 
                # words contain <eow> token at the end of word (final char)
                l, probs = generate_word(hs, hidden_generator, t, device=device)
                loss += l


            # @TODO: one could include a word level predictor here 

            # here one could split batch apart so that only single words are predicted
            # one could parallelise the generation process (problem: don't want to detach it from current graph) 
 
            loss.backward()
            optimizer.step()
            beginning_char = end_char
            end_char = beginning_char + bs
            #data_char_target_word = torch.transpose(data_char_target_word, 0,1)
            
            hidden_state = repackage_hidden(hidden_state)
            hidden_char = repackage_hidden(hidden_char)
            hidden_generator = repackage_hidden(hidden_generator)



best_val_loss = None
for epoch in range(1, epochs+1):
    train(train_d)
    val_loss = evaluate(val_d)
    #print('VAL LOSS', val_loss)
    if not best_val_loss or val_loss < best_val_loss:
        with open(model_path, 'wb') as f:
            torch.save(model, f)
            best_val_loss = val_loss
            print(best_val_loss)
    else:
        lr /= 4.0
"""
hidden_states = []
for i in len(input):
  _, (h, c) = self.lstm(embeddings[i], (h, c))
  # do whatever other manipulations you need to do on h and c
  hidden_states.append(h)
hidden_state_tensor = torch.stack(hidden_states)
"""