from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(128, input_dim=784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='sgd', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./data/", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

model.fit(train_images, train_labels, epochs=100, batch_size=128)
score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
