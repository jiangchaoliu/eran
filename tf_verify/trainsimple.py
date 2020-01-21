# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset


model_name = 'acnn3t8.h5'
x_file = 'cnn3t8.csv'
max_num = 200

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
model = Sequential()
model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)
# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))
x_test = x_test[:300]
y_test = y_test[:300]
# Craft adversarial samples with FGSM
epsilon = .1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test)
# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
print(preds.shape)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))
#save the model
model.save(model_name)
#save x and cresponding adv in eran format



x_savel = np.zeros(28*28+1)
x_advl = np.zeros(28*28+1)
num = 0
for i in range(len(x_test)):
    if num > max_num:
        break
    x = x_test[i].reshape(1,28,28,1)
    y = np.argmax(y_test[i])
    pred = np.argmax(classifier.predict(x))
    if y != pred:
        x_adv_save = x.copy().flatten()
        x_adv_save = np.concatenate((np.array([pred]),x_adv_save))
        x_advl = np.row_stack((x_advl,x_adv_save))
    else:
        x_save = x.copy().flatten()
        x_save = np.concatenate((np.array([y]),x_save))
        x_savel = np.row_stack((x_savel,x_save))
    num += 1
print(num)
x_savel=x_savel[1:]
x_advl =x_advl[1:]
np.savetxt(x_file,x_savel,fmt='%.4f',delimiter=',')

#save x and cresponding adv in eran format
