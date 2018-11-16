import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, TimeDistributed
from keras.models import Model
from keras.backend import clear_session
from functions import *
import string
import random


def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt

def normalize_text(text, valid_chars, replace_chars):
    text = text.lower()
    for char, replacewith in replace_chars.items():
        text = text.replace(char, replacewith)
    text = ''.join([c for c in text if c in valid_chars])
    return text.replace('  ', ' ')

def build_vocab_map(valid_chars):
    # Reserve 0 for "blank" character
    vocab = dict([(ch, idx) for idx, ch in enumerate([''] + sorted(valid_chars))])
    inv_vocab = dict([(idx, ch) for ch, idx in vocab.items()])
    return vocab, inv_vocab

def sample_text(text, sample_length, shuffle=True):
    i = 0
    max_i = len(text) - sample_length
    i_vals = list(range(max_i))
    while True:
        if shuffle:
            random.shuffle(i_vals)
        for i in i_vals:
            inputs = text[i:i+sample_length]
            targets = text[i+1:i+sample_length+1]
            if i + sample_length >= len(text):
                inputs = inputs[:len(text) - sample_length - 1]
                targets = targets[:len(text) - sample_length]
            yield (inputs, targets)

def encode_text(text, vocab, max_len):
    return [vocab.get(text[i], 0) if i < len(text) else 0 for i in range(max_len)]

def one_hot_elem(index, vocab):
    a = np.zeros(len(vocab))
    a[index] = 1
    return a

def one_hot(encoded, vocab):
    return [one_hot_elem(idx, vocab) for idx in encoded]

def sample_batch(text, sample_length, batch_size, vocab, shuffle=True):
    sampler = sample_text(text, sample_length, shuffle)
    while(True):
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            inputs, targets = next(sampler)
            batch_inputs.append(encode_text(inputs, vocab, sample_length))
            batch_targets.append(one_hot(encode_text(targets,vocab, sample_length), vocab))
        yield np.array(batch_inputs), np.array(batch_targets) 

        
def likely_index(preds, temperature=1.0):
    #return np.argmax(preds)
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, vocab, inv_vocab, max_len, len_to_generate):
    seed_text = seed_text[:max_len]
    output = [c for c in seed_text]
    for i in range(len_to_generate):
        seed_input = np.array([encode_text(seed_text, vocab, max_len)])
        idx_after_seed = len(seed_text) - 1
        #print(seed_text)
        predicted = model.predict(seed_input)[0]
        next_char = inv_vocab[likely_index(predicted[idx_after_seed], temperature=0.3)]
        output.append(next_char)
        if len(seed_text) < max_len:
            seed_text = seed_text + next_char
        else:
            seed_text = seed_text[1:] + next_char
    return ''.join(output)
        
    
    
def build_lstm_model(max_text_len, num_lstms, lstm_hidden_dim, embedding_dim, vocab, optimizer=None):
    max_vocab_index = max([v for k,v in vocab.items()])
    vocab_size = len(vocab)
    inp = Input(shape=(max_text_len,))
    x = Embedding(input_dim=max_vocab_index, output_dim=embedding_dim, input_length=max_text_len)(inp)
    for _ in range(num_lstms):
        x = LSTM(lstm_hidden_dim, return_sequences=True)(x)
        x = Dropout(0.5)(x)
    x = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='adam' if optimizer is None else optimizer, metrics=['accuracy'])
    return model