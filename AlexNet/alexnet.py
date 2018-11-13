# -*- coding: utf-8 -*-
"""

This module demonstrates the AlexNet NN architecture.
0) Relu activations instead of Tanh.
1) Dropout instead of regularisation to avoid overfitting (0.5).
2) Overlap pooling to reduce network size.

convolutional layers - 5
fully connected      - 3
* apply Relu after every convolutional and fully connected layer
* apply Dropout after 1st, 2nd fully connected layer

input: 227 x 227 (images are 224 x 224, add 2 padding)
output: classification

Example Usage:
    $ python alexnet.py

Todo:
    * Todos here

"""

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils import np_utils
import numpy as np
import random
import sys
import io
import argparse

if __name__ == '__main__':

    # TODO: Fill in the dimensions per the paper
    model = Sequential()
    model.add(Conv2D(96, 11, strides=4, input_shape=(224, 224), activation=Activation('relu')))
    model.add(Conv2D())
    model.add(Conv2D())
    model.add(Conv2D())
    model.add(Conv2D())
    model.add(Dense())
    model.add(Dense())
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.summary()




