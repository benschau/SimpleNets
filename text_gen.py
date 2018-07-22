from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random 
import sys
import io

chars = sorted(list(set(text)))
print('total chars: {}'.format(len(chars)))

char_indices = {}
indices_char = {}
for i, c in enumerate(chars):
    char_indices[c] = i
    indices_char[i] = c

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences: {}', len(sentences))

print('vectorize...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1

    y[i, char_indices[next_chars[i]]] = 1

print('build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)








class Corpus:
    def __init__(self, _path):
        path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt') 
        with io.open(path, encoding='utf-8') as f:
            self.text = f.read().lower()

        print('corpus len: {}'.format(len(text)))
        clean()

    def clean(self):
        self.text = self.text.decode('utf-8').encode('ascii', 'ignore') 
        words = self.text.split() 

    def vectorize(self):
        pass

class Model:
    def __init__(self):
        pass

    def build_graph(self):
        pass

    def create_test_sample(self):
        pass

    def epoch_end(self):
        pass

    def train(self):
        pass
