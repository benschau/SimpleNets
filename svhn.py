from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from progress.bar import Bar

tf.logging.set_verbosity(tf.logging.INFO)

# I want to be able to do something like this for SVHN....
# mnist = input_data.read_data_sets("./data/", one_hot=True)
# train_images = mnist.train.images
# train_labels = mnist.train.labels
# test_images = mnist.test.images
# test_labels = mnist.test.labels

class svhn:
  
    def read_data_sets(self, path, one_hot):
        pass

    class test: 
        def __init__(self):
            pass

    class train: 
        def __init__(self):
            pass

        def process_train(self):
            pass


svhn = svhn.read_data_sets("./data/", one_hot=True)
train_images = svhn.train.images
train_labels = svhn.train.labels
test_images = svhn.test.images
test_labels = svhn.test.labels


