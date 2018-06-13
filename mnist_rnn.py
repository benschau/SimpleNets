import tensorflow as tf
import numpy as np
from progress.bar import Bar
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

input_vec_size = lstm_size = 28
time_step = 28

batch = 128
test = 256

def lstm(X, weights, biases, lstm_size):
    # transform input into processable tensors
    _Xtranspose = tf.transpose(X, [1, 0, 2])
    _Xreshape = tf.reshape(_Xtranspose, [-1, lstm_size]) 
    _Xsplit = tf.split(_Xreshape, time_step, 0) # => 28 arrays of [batch_size, input_vec_size]
    
    # or, LSTMCell; either works, LSTMCell has a more advanced implementation behind it.
    # according to the documentation.
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    
    # create RNN specified by lstm
    outputs, _states = rnn.static_rnn(lstm, _Xsplit, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases, lstm.state_size 


mnist = input_data.read_data_sets("data/", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

train_images.reshape(-1, 28, 28)
test_images.reshape(-1, 28, 28)

X = tf.placeholder("float32", [None, 28, 28])
Y = tf.placeholder("float32", [None, 10])

weights = tf.Variable(tf.random_normal([lstm_size, 10], stddev=0.01))
biases = tf.Variable(tf.random_normal([10], stddev=0.01))

outX, state_size = lstm(X, weights, biases, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outX, labels=Y))
train = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(outX, 1)

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True # let GPU allocate as much memory it needs as it runs

with tf.Session(config=conf) as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        # some color, so I'll at least have added something original to this:
        print("Epoch {}: ".format(i))
        bar = Bar('Training: ', max=(len(train_images) / batch))
        
        for start, end in zip(range(0, len(train_images), batch), range(batch, len(train_images) + 1, batch)):
            sess.run(train, feed_dict={X: train_images[start:end], Y: train_labels[start:end]})
            bar.next()

        bar.finish()

        test_indices = np.arange(len(test_images))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test]
        
        print("test accuracy: {}".format(np.mean(np.argmax(test_labels[test_indices], axis=1) == sess.run(predict, feed_dict={X: test_images[test_indices]}))))

        # maybe? train accuracy is how well the network does against the data it's being trained against
        # rnn caveat: should i use the same test_indices? or create another random range for train...
        print("train accuracy: {}".format(np.mean(np.argmax(train_labels[test_indices], axis=1) == sess.run(predict, feed_dict={X: train_images[test_indices]}))))




