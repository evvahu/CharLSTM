import torch
import torch.nn as nn
import torch.utils.data.dataloader
from utilsy import get_char_input


class Encoder(nn.Module):
    def __init__(self, dropout, ninp, nhid, nlayers, ntoken, params_char = '', rnn_type = 'LSTM'):
        super(Encoder, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.encoder = nn.Embedding(ntoken, ninp)
        
        self.charEncoder = CharEncoder(*params_char)

        rnn_dim = ninp + params_char[1]
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_dim, nhid, nlayers, batch_first=False, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu', 'RNN_ELU': 'elu', 'RNN_SELU':'selu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_dim, nhid, nlayers, nonlinearity=nonlinearity, batch_first=False, dropout=dropout)
          
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, word_input, char_input, rnn_hidden, hidden_char):
        #print('CHAR', char_input.shape, char_hidden.shape)
        if torch.cuda.is_available():
            word_input = word_input.cuda() #to(device)
            char_input = char_input.cuda() #.to(device)

        #hidden_char.to(device)
        _, hidden_char = self.charEncoder(char_input, hidden_char) # take last hidden state of character LSTM 
        
        emb_word = self.encoder(word_input)
        #emb_word = torch.flatten(emb_word, )
        #hidden_char[0].view(emb_word.shape[0],-1)), 1 OOOOLD
        #hidden_char_ref = hidden_char[0].squeeze()
        hidden_char_ref = hidden_char[0].squeeze()
        emb_concat = self.drop(torch.cat((emb_word, hidden_char_ref), 1)) # hidden_char[0] is hidden state (1 is cell state)
        emb_concat = emb_concat.unsqueeze(0)#.unsqueeze(0) 
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

    def __init__(self, tokensize, ninp, nhid, dropout, nlayers=1, rnn_type='LSTM'):
        super(CharEncoder, self).__init__()
        self.encoder = nn.Embedding(tokensize, ninp, padding_idx =0)
        print('sizes of encoder ', tokensize, ninp )
        self.decoder = nn.Linear(nhid, tokensize)
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.nhid = nhid
        self.drop = nn.Dropout(dropout)

        if rnn_type in ['LSTM', 'GRU']:
            print('RNN TYPE', rnn_type)
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers,batch_first=True, dropout=dropout)
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
        #input.to(device)
        if torch.cuda.is_available():
            input = input.cuda()#to(device)
            self.encoder = self.encoder.cuda() #to(device)
        #print('input shape', input.shape)
        emb = self.drop(self.encoder(input))
        #emb = self.drop(emb)
        #print('emb and hidden shape', emb.shape, hidden.shape)
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
        """
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()
        """

class CharGenerator(nn.Module):
    def __init__(self, hl_size, emb_size, nchar, nhid, nlayers, dropout,padding_id=0, rnn_type = 'LSTM'):
        """
        hl_size: size of hidden layer of main LSTM
        emb_size: size of character embedding 
        nchar: number of characters 
        """
        super(CharGenerator, self).__init__()
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
        self.encoder = nn.Embedding(nchar+1, emb_size, padding_idx=padding_id)
        self.decoder = nn.Linear(nhid, nchar)
        self.drop = nn.Dropout(dropout)
        #self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    def forward(self, input, hidden_lstm, hidden):
        """
        last_char: index of previously predicted character
        output: trained to predict the next char 
        """
        if torch.cuda.is_available():
            input = input.cuda()
        input = self.encoder(input)
        #hidden_lstm = hidden_lstm[1].squeeze()
        input_cat = torch.cat((input, hidden_lstm), 0) # needs to be right dim 
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