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
        
        #self.charEncoder = CharEncoder(*params_char)

        #rnn_dim = ninp + params_char[1]
        rnn_dim = ninp
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_dim, nhid, nlayers, batch_first=False, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu', 'RNN_ELU': 'elu', 'RNN_SELU':'selu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_dim, nhid, nlayers, nonlinearity=nonlinearity, batch_first=False, dropout=dropout)
          
        #self.decoder = nn.Linear(nhid, ntoken)
        self.decoder = CharGenerator(*params_char)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, word_input, char_input, rnn_hidden, hidden_char, device):
        device2 = 'cuda:{}'.format(torch.cuda.current_device())
        print('current device: {}, {}'.format(device, device2))
        #self.encoder = self.encoder.to(device)
        #self.rnn = self.rnn.to(device)
        #self.decoder = self.decoder.to(device)
        device = torch.device(device)
        word_input = word_input.to(device)
        char_input = char_input.to(device)
        rnn_hidden = [rnn_hidden[0].to(device), rnn_hidden[1].to(device)]
        hidden_char = [hidden_char[0].to(device), hidden_char[1].to(device)]
        #rnn_hidden = rnn_hidden.to(device)
        #hidden_char = hidden_char.to(device)
        print('device:', word_input.get_device(), char_input.get_device())
        print('in forward size: {}'.format(word_input.shape))
        #hidden_char.to(device)
        #_, hidden_char = self.charEncoder(char_input, hidden_char) # take last hidden state of character LSTM 
        hs_main = []
        hs_chars = []
        outputs = []
        #self.encoder.to(device)
        for id in range(word_input.shape[1]-1):
            #print('in forward', word_input[:,id].shape)
            emb_word = self.encoder(word_input[:,id]).unsqueeze(0)
            #emb_word = torch.flatten(emb_word, )
            #hidden_char[0].view(emb_word.shape[0],-1)), 1 OOOOLD

            #hidden_char_ref = hidden_char[0].squeeze()
            #emb_concat = self.drop(torch.cat((emb_word, hidden_char_ref), 1)) # hidden_char[0] is hidden state (1 is cell state)
            #emb_concat = emb_concat.unsqueeze(0)#.unsqueeze(0) 
            #output, hidden = self.rnn(emb_concat, rnn_hidden)
           # print('emb word shape: {}'.format(emb_word.shape))
            #output, hidden = self.rnn(emb_word, rnn_hidden)
            output, hidden = self.rnn(emb_word)
            current_char = char_input[:,id, :].long()
            #output_decoder, hidden_char = self.decoder(current_char, hidden[0][1].squeeze()) #char input has to be n+1
 
            output_decoder, hidden_char = self.decoder(current_char, hidden[0][1].squeeze(), hidden_char) #char input has to be n+1
            outputs.append(output_decoder.view(output_decoder.shape[0], output_decoder.shape[2], -1))
            #hs_main.append(hidden)
            #hs_chars.append(hidden_decoder)
            #output = self.drop(output)
            #decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        
        #return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, hidden_char
        return outputs, hidden, hidden_char #torch.stack(hs_main), torch.stack(hs_chars)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
       # self.decoder.bias.data.fill_(0)
       # self.decoder.weight.data.uniform_(-initrange, initrange)


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
        self.encoder = nn.Embedding(tokensize, ninp, padding_idx =0)
        self.decoder = nn.Linear(nhid, tokensize)
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
        emb = self.drop(self.encoder(input))
        #hidden_lstm = hidden_lstm[1].squeeze()
        #hidden_lstm = hidden_lstm.repeat()
        #print('in generator emb and hidden', emb.shape, hidden_lstm.shape)
        hidden_lstm = torch.cat([hidden_lstm]*emb.shape[1]).reshape(emb.shape[0], emb.shape[1],-1)
        input_cat = torch.cat((emb, hidden_lstm), 2) # hidden_lstm need to be inserted for each word 
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