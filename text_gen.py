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

def get_sample():
    pass

DATA_DIR = "data/donquixote.txt"

# prep training data
data = open(DATA_DIR, 'r').read().lower()
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

fpath = 'checkpoints/donquixote-{epoch:02d}-{loss:.4f}.hdf5'
ckpt = ModelCheckpoint(fpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [ckpt]

model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)
