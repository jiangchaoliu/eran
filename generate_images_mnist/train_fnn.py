# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset



import sys

assert(len(sys.argv) > 3)


layers = int(sys.argv[1])
neurons = int(sys.argv[2])
types = sys.argv[3]

model_name = ''
if types == 'sigmoid':
        model_name = 'snet'

if types == 'tanh':
        model_name = 'tnet'

if types == 'relu':
        model_name = 'rnet'

model_name = model_name+sys.argv[1]+'t'+sys.argv[2]+'.h5'
        
# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))


model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
for i in range(layers):
        model.add(Dense(neurons, activation=types))
        
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)
# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))
model.save(model_name)

