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

DATA_DIR = "data/trump_speeches.txt"

# prep training data
# using ryanmcdermott's trump speeches dataset, https://github.com/ryanmcdermott/trump-speeches
data = open(DATA_DIR, 'r')
