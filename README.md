

# Character-level Language Model

The important files are the following two:

- model.py: contains code for all models
    - Encoder: main LSTM language model
    - CharEncoder: to produce character-based encoding of incoming word
    - CharGenerator: to predict characters of next word in the sequence 
- run.py: main file to train and evaluate 

<<<<<<< HEAD

The main idea of the model is a language modelling task (i.e. predict next word given the previous word(s)), but instead of predicting the index of the next word, we want to predict the character string of the next word. 
The main model (*Encoder*) takes as input a concatenation of the embedding of the word (simple look-up matrix) and the last_hidden state of a character-level LSTM (*CharEncoder*). 
The generator (*CharGenerator*) predicts the string of the next word (t+1) based on a concatenation of the hidden state of the word from the main LSTM at time point t and the character at time point n in the CharGenerator. 
In order to retrieve the hidden states of each word in the sequence, we have to loop over the words in the sequence. Thus, at each time point t, we generate the string of the next word (with method *generate_bs*). We do this over the whole batch one word at a time (this is possible because each sequence contains same number of words and is not padded). Thus the loss that is calculated over the characters of a word at time point n of the whole batch (I don't know whether this is ideal). The words are restricted to be a certain maximum length and the ones that are shorter are padded with 0s. Each word contains an \<eow\> character.  
=======
Description of model:

- The main idea of the model is a language modelling task (i.e. predict next word given the previous word(s)), but instead of predicting the index of the next word, we want to predict the character string of the next word. 
- The main model (*Encoder*) takes as input a concatenation of the embedding of the word (simple look-up matrix) and the last_hidden state of a character-level LSTM (*CharEncoder*). 
- The generator (*CharGenerator*) predicts the string of the next word (t+1) based on a concatenation of the hidden state of the word from the main LSTM at time point t and the character at time point n in the CharGenerator. 
- In order to retrieve the hidden states of each word in the sequence, we have to loop over the words in the sequence. Thus, at each time point t, we generate the string of the next word (with method *generate_bs*). We do this over the whole batch one word at a time (this is possible because each sequence contains same number of words and is not padded). Thus the loss that is calculated over the characters of a word at time point n of the whole batch (I don't know whether this is ideal). The words are restricted to be a certain maximum length and the ones that are shorter are padded with 0s. Each word contains an \<eow\> character.  
>>>>>>> 5cc600c80047837401fb8dff4fcddcea48b48efc
