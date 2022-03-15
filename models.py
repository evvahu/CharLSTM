import torch
import torch.nn as nn
import torch.utils.data.dataloader
from utilsy import get_char_input
class WordEncoder(nn.Module):

    def __init__(self, dropout, ninp, nhid, nlayers, ntoken, params_char = '', params_morph = '', rnn_type = 'LSTM', models='CW'):
        super(WordEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        assert models == 'CW' or models == 'W' or models == 'C' or models == 'CMW' or models == 'CM' or models == 'M', "wrong model option indicated, has to be CW or W or C"
        self.models = models
        self.encoder = nn.Embedding(ntoken, ninp)
        if 'C' in self.models: #self.models == 'CW' or self.models == 'C':
            self.charEncoder = CharMorphEncoder(*params_char)
        if 'M' in self.models:
            self.morphEncoder = CharMorphEncoder(params_morph)
        #elif self.models == 'W':
        if models == 'CW':
            rnn_dim = ninp + (params_char[1])
        elif models == 'C':
            rnn_dim = params_char[1]
        elif models == 'CM':
            rnn_dim =  ninp + params_char[1]
        elif models == 'CMW':
            rnn_dim = ninp + params_char[1] + params_morph[1]
        elif models == 'CM':
            rnn_dim = params_char[1] + params_morph[1]
        elif models == 'M':
            rnn_dim = params_morph[1]
        elif models == 'W':
            rnn_dim = ninp
        else:
            rnn_dim = ninp
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_dim, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, word_input, char_input, morph_input, rnn_hidden, char_hidden, morph_hidden):
        if self.models == 'W':
            emb = self.drop(self.encoder(word_input))
        elif self.models == 'CW':
            #emb_char = self.charEncoder(char_input, char_hidden) #
            #emb_char = emb_char.view(emb_char.shape[1], emb_char.shape[0]* emb_char.shape[2])
            _, hidden_char = self.charEncoder(char_input, char_hidden) # take last hidden state of character LSTM 
            print(hidden_char[0].shape)
            emb_word = self.encoder(word_input)
            #emb_char = emb_char.reshape(emb_word.shape[0], emb_word.shape[1], -1)
            #emb_char = hidden_char[0].reshape(emb_word.shape)
            #print(emb_word.shape, emb_char.shape)
            #emb = self.drop(torch.cat((emb_word, emb_char), dim=2)) # concatenate
            emb = self.drop(torch.cat(emb_word, hidden_char))
        elif self.models == 'C':
            emb = self.drop(self.charEncoder(char_input))
        elif self.models == 'CMW':
            _, hidden_morph = self.morphEncoder(morph_input, morph_hidden)
            _, hidden_char = self.charEncoder(char_input, char_hidden)
            emb_word = self.encoder(word_input)
            emb = self.drop(torch.cat(emb_word, hidden_morph, hidden_char)) 
        elif self.models == 'CM':
            _, hidden_morph = self.morphEncoder(morph_input, morph_hidden)
            _, hidden_char = self.charEncoder(char_input, char_hidden)
            emb = self.drop(torch.cat(hidden_morph, hidden_char)) 

        output, hidden = self.rnn(emb, rnn_hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()

class CharMorphEncoder(nn.Module):

    def __init__(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
        super(CharMorphEncoder, self).__init__()
        self.encoder = nn.Embedding(tokensize+1, ninp, padding_idx =0)
        self.decoder = nn.Linear(nhid, tokensize)
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.nhid = nhid
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        return output, hidden
        #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()








