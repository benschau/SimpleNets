from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from progress.bar import Bar

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("./data/", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

_x = tf.placeholder("float32", [None, 784])
_y = tf.placeholder("float32", [None, 10])

weights = {
    "w_h": tf.Variable(tf.random_normal([784, 625], stddev=0.01)),
    "w_o": tf.Variable(tf.random_normal([625, 10], stddev=0.01))
}

mlp = tf.nn.sigmoid(tf.matmul(_x, weights['w_h']))
model = tf.matmul(mlp, weights['w_o'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=_y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict = tf.argmax(model, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(100):
        # some color, so I'll at least have added something original to this:
        print("Epoch {}: ".format(i))
        bar = Bar('Training: ', max=(len(train_images) / 128))
        
        for start, end in zip(range(0, len(train_images), 128), range(128, len(train_images) + 1, 128)):
            sess.run(train, feed_dict={_x: train_images[start:end], _y: train_labels[start:end]})
            bar.next()

        bar.finish()
        
        print("test accuracy: {}".format(np.mean(np.argmax(test_labels, axis=1) == sess.run(predict, feed_dict={_x: test_images}))))

        # maybe? train accuracy is how well the network does against the data it's being trained against
        print("train accuracy: {}".format(np.mean(np.argmax(train_labels, axis=1) == sess.run(predict, feed_dict={_x: train_images}))))


