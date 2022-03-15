

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time
import fasttext
import torch
import torch.nn as nn

from dictionary_corpus import Corpus
import models
from lm_argparser import lm_parser
from utils import repackage_hidden, get_batch, batchify, ids_to_embs

parser = argparse.ArgumentParser(parents=[lm_parser],
                                 description="Basic training and evaluation for RNN LM")

args = parser.parse_args()
# comment
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(args.log)])
logging.info(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

logging.info("Loading data")
start = time.time()
corpus = Corpus(args.data, args.morph_path)
logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)

logging.info("Batchying..")
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args.cuda)
val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
test_data = batchify(corpus.test, eval_batch_size, args.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")
# args.type_model
model = models.WordEncoder() #include arguments  #RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.type_encoder, args.path_to_model, args.dropout, args.tied)
if args.cuda:
    model.cuda()


###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
                #if torch.cuda.is_available():
                #data.to(device)
                #print('is data on cuda in evaluate?', data.is_cuda)
            #> output has size seq_length x batch_size x vocab_size
            output, hidden = model(data, hidden)   #h_n: tensor of shape (D∗num_layers,N,Hout)(D * \text{num\_layers}, N, H_{out})(D∗num_layers,N,Hout​) containing the final hidden state for each element in the batch
            #> output_flat has size num_targets x vocab_size (batches are stacked together)
            #> ! important, otherwise softmax computation (e.g. with F.softmax()) is incorrect
            output_flat = output.view(-1, ntokens)
            #output_candidates_info(output_flat.data, targets.data)
            total_loss += len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)



def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    criterion_char = ''
    criterion_morph = ''
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)   
        # truncated BPP
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        if args.type_model == 'CW':
            loss = criterion(output.view(-1, ntokens), targets)
        elif args.type_model == 'C':
            char_loss = criterion_char() 
            word_loss = criterion(output.view(-1, ntokens), targets)
            loss = char_loss + word_loss
        else:
            loss = criterion(output.view(-1, ntokens), targets)

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #for name, param in model.named_parameters():
            #if param.requires_grad:
                #print(name)#, param.data)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train()

        val_loss = evaluate(val_data)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
logging.info('=' * 89)

