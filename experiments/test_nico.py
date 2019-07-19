# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:48:29 2017

@author: thomen
"""
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

outfile = 'DF_ResNet50_VOC2007.npz'
learning_rate = 0.1
nb_epoch = 20
batch_size = 32

npzfile = np.load(outfile)

X_train = npzfile['X_train']
Y_train = npzfile['Y_train']

X_test = npzfile['X_test']
Y_test = npzfile['Y_test']

print("données chargées \n X_train=", X_train.shape, "Y_train=", Y_train.shape, " X_test=", X_test.shape, "Y_train=", Y_test.shape)

model = Sequential()
model.add(Dense(20, input_dim=2048, name='fc1', activation='sigmoid'))
model.summary()


sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])

model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s TEST: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s TEST: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
AP_train = np.zeros(20)
AP_test = np.zeros(20)

for c in range(20):
    AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
    AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])

print("MAP TRAIN =%.2f", AP_train.mean()*100)
print("MAP TEST =%.2f", AP_test.mean()*100)

# PERFS
# 10 epochs: 81.3
# 20 epochs: 82.54
# 30 epochs: 82.78
# 50 epochs: 82.75
# 100 eopchs: 82.31
