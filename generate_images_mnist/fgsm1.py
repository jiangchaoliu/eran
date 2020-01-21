# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks import FastGradientMethod
from art.attacks import CarliniL2Method
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from keras.models import load_model

import sys

assert(len(sys.argv) > 2)


model_name = sys.argv[1]
adv_file = sys.argv[2]
max_num = 100

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
model = load_model(model_name)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))
# Craft adversarial samples with FGSM
x_test = x_test[:500]
y_test = y_test[:500]
epsilon = .1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier,eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test)
# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
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
        continue
    x_adv = x_test_adv[i].reshape(1,28,28,1)
    adv_pred = np.argmax(classifier.predict(x_adv))
    print(adv_pred,y)
    if adv_pred == y:
        continue
    x_save = x.copy().flatten()
#    x_save *= 255
    x_save = np.concatenate((np.array([pred]),x_save))
    x_adv_save = x_adv.copy().flatten()
#    x_adv_save *= 255
    x_adv_save = np.concatenate((np.array([adv_pred]),x_adv_save))
    x_savel = np.row_stack((x_savel,x_save))
    x_advl = np.row_stack((x_advl,x_adv_save))
    num += 1
print(num)
x_asvl =x_advl[1:]
np.savetxt(adv_file,x_advl,fmt='%.4f',delimiter=',')

