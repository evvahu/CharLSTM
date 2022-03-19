import torch
import torch.nn as nn
import torch.utils.data.dataloader
from utilsy import get_char_input


class Encoder(nn.Module):
    def __init__(self, dropout, ninp, nhid, nlayers, ntoken, device, params_char = '', rnn_type = 'LSTM'):
        super(Encoder, self).__init__()
        self.device = device
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.encoder = nn.Embedding(ntoken, ninp)
        
        self.charEncoder = CharEncoder(*params_char, dev = self.device)
        self.charEncoder.to(self.device)
        rnn_dim = ninp + params_char[1]
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

    def forward(self, word_input, char_input, rnn_hidden, hidden_char):
        #print('CHAR', char_input.shape, char_hidden.shape)
        if self.device == 'cuda:0':
            word_input = word_input.cuda()
            char_input = char_input.cuda()
        _, hidden_char = self.charEncoder(char_input, hidden_char) # take last hidden state of character LSTM 
        
        emb_word = self.encoder(word_input)
        #emb_word = torch.flatten(emb_word, )
        emb_concat = self.drop(torch.cat((emb_word, hidden_char[0].view(emb_word.shape[0],-1)), 1)) # hidden_char[0] is hidden state (1 is cell state)
        emb_concat = torch.unsqueeze(emb_concat, 0)
        #print('rnn hidden', rnn_hidden.shape)
        output, hidden = self.rnn(emb_concat, rnn_hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, hidden_char

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

class CharEncoder(nn.Module):

    def __init__(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM', dev='cpu'):
        super(CharEncoder, self).__init__()
        self.device = dev
        self.encoder = nn.Embedding(tokensize, ninp, padding_idx =0)
        self.decoder = nn.Linear(nhid, tokensize)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.nhid = nhid
        self.drop = nn.Dropout(dropout)

        if rnn_type in ['LSTM', 'GRU']:
            print('RNN TYPE', rnn_type)
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            print('wrong type')
            #try:
            #    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            #except KeyError:
            #    raise ValueError( """An invalid option for `--model` was supplied,
            #                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            #self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.init_weights()
    def forward(self, input, hidden):

        input = torch.tensor(input)
        if self.device == 'cuda:0':
            input = input.cuda()
      
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


class CharGenerator(nn.Module):
    def __init__(self, hl_size, emb_size, nchar, nhid, nlayers, dropout, device, rnn_type = 'LSTM'):
        """
        hl_size: size of hidden layer of main LSTM
        emb_size: size of character embedding 
        nchar: number of characters 
        """
        super(CharGenerator, self).__init__()
        self.device = device
        ninp = hl_size + emb_size
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.encoder = nn.Embedding(nchar+1, emb_size)
        self.decoder = nn.Linear(nhid, nchar)
        self.drop = nn.Dropout(dropout)
        #self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    def forward(self, input, hidden_lstm, hidden):
        """
        last_char: index of previously predicted character
        output: trained to predict the next char 
        """
        if self.device == 'cuda:0':
            input = input.cuda()
        #last_char.to_device(self.device)
        input = self.encoder(input)
        hidden_lstm = hidden_lstm[0]
        input_cat = torch.cat((input, hidden_lstm.squeeze()), 0) # needs to be right dim
        input_cat = torch.unsqueeze(input_cat, 0).unsqueeze(0)
        input_cat = self.drop(input_cat)
        output, hidden = self.rnn(input_cat, hidden)
        output = self.drop(output)
        output = self.decoder(output)
        #output = self.softmax(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)