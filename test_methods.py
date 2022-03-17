import torch
from dict_old import Dictionary
from run import generate_word
from utilsy import char_repr
import re 
def generate_sentence(model, generator, sents, dictionary):
    hidden_generator = generator.init_hidden(1)
    hidden_char = model.CharEncoder.init_hidden(1)
    hidden_word = model.init_hidden(1)
    last_idx = dictionary.word2idx['<eow>']
    with torch.no_grad():
        for sent in sents:
            for i, word in enumerate(sent):
                word_str = dictionary.idx2word[word]
                char = char_repr(word_str, dictionary.char2idx, len(word_str)+1, '<eow>')     #word, chardict, max_l, eow):
                out, hidden = model(word, char, hidden_word, hidden_char)
                b = hidden[0]
                if i > (len(sent)-1):
                    target_w = dictionary.word2idx['.<eos>'] # without dot or with?
                else:
                    target_w = sent[i+1]
                _, avg_probs, word_str = generate_word(hidden[0], hidden_generator, target_w, last_idx, len(word_str)+1, device = 'cpu')

#self, word_input, char_input, rnn_hidden, hidden_char):

def indexify(dictionary, word):
    if word not in dictionary.word2idx:
        print("Warning: {} not in vocab".format(word))
        count +=1
    return dictionary.word2idx[word] if word in dictionary.word2idx else dictionary.word2idx["<unk>"]

def evaluate_sentence(model_p, generator_p, data_p, sent_p): 
    """
    model_p: path where model is stored
    generator_p: path where generator is stored
    sent_p: path to sentences 
    """
    corpus = Dictionary(data_p)
    model = torch.load(model_p)
    generator = torch.load(generator_p)
    # for each word in sentence: get the probabilities of the target word 
    sentences = []
    with open(sent_p, 'r') as rf:
        next(rf)
        for l in rf:
            l = l.strip()
            line = l.split('\t')
            indexed = [indexify(corpus.dictionary, w) for w in line[-1].strip('.|,')]
            sentences.append(indexed)
    generate_sentence(model, generator, sentences, corpus.dictionary)


if __name__ == '__main__':
    model_path = ''
    generator_path = ''
    data_path = ''
    sent_path = ''
    evaluate_sentence(model_path, generator_path, data_path, sent_path) 




