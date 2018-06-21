""" 
RNN Encoder 

An RNN neural network model based off architecture as explained in Learnining Phrase Representations using 
RNN Encoder-Decoder for Statistical Machine Translation (Cho et. al).

Author: Benson Chau
"""

import importlib
import tensorflow as tf
import numpy as np

data_module = importlib.import_module('LanguageData')
LanguageData = getattr(data_module, 'LanguageData')

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
max_num_input = 50
timesteps = 28 # timesteps 
num_hidden = 1000 # hidden layer num of features

data = LanguageData()

print(data.english)
print(data.french)
