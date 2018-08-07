from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils import np_utils
import numpy as np
import random 
import sys
import io
import argparse

def get_sample(model, ckpt_path, ind_to_char, seed): 
    model.load_weights(ckpt_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    chars_size = len(ind_to_char)
    print("Generating 1000 character sample...")
    print("Seed: ")
    for ind in seed:
        print(ind_to_char[ind], end='', flush=True)
   
    for i in range(1000):
        X = np.reshape(seed, (1, len(seed), 1))
        print(X)
        X = X / float(chars_size)
        pred = model.predict(X, verbose=0)
        index = np.argmax(pred)

        result = ind_to_char[index]
        print(result, end='', flush=True)
        
        seed.append(result)
        seed = seed[1:len(seed)]

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Simple training and sampling on a double stacked LSTM.')
    parser.add_argument('--sample', type=str, help='randomly select a character and produce a string from the given checkpoint path.')

    args = parser.parse_args()

    DATA_DIR = "data/donquixote.txt"

    # prep training data
    data = open(DATA_DIR, 'r').read().lower()
    data = data.encode('utf-8').decode('unicode-escape')
    chars = list(set(data))
    chars_size = len(chars)

    print('total characters: {}'.format(len(data)))
    print('total unique characters/vocab: {}'.format(chars_size))

    index_char = {}
    char_index = {}
    for index, char in enumerate(chars):
        index_char[index] = char
        char_index[char] = index

    seq_len = 100
    seqX = []
    seqY = []
    for i in range(len(data) - seq_len):
        seq_in = data[i:i + seq_len]
        seq_out = data[i + seq_len]
        seqX.append([char_index[char] for char in seq_in])
        seqY.append(char_index[seq_out])

    patterns_size = len(seqX)

    print('total patterns: {}'.format(patterns_size))
    X = np.reshape(seqX, (patterns_size, seq_len, 1))
    X = X / float(chars_size)
    Y = np_utils.to_categorical(seqY)

    # build the model:
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.summary()
    
    if args.sample is not None:
        print("Getting model from {}...".format(args.sample))
        seed = seqX[np.random.randint(0, len(seqX))]
        get_sample(model, args.sample, index_char, seed)
        sys.exit()

    fpath = 'checkpoints/donquixote-{epoch:02d}-{loss:.4f}.hdf5'
    ckpt = ModelCheckpoint(fpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [ckpt]

    model.fit(X, Y, epochs=10, batch_size=128, callbacks=callbacks_list)
