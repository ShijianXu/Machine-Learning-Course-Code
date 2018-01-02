# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:47:54 2017
Machine Learning
@author: XU Shijian 141220120
"""
import numpy as np
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

train_data = np.genfromtxt('train_data.csv',delimiter=',')
train_targets = np.genfromtxt('train_targets.csv')
test_data = np.genfromtxt('test_data.csv',delimiter=',')

nb_classes = 10
train_targets = np_utils.to_categorical(train_targets, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(400,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit(train_data, train_targets,
          batch_size=128, epochs=60,
          verbose=1)

predicted_classes = model.predict_classes(test_data)
row = []
for item in predicted_classes:
    row.append((item,))
csvFile = open("test_predictions_library.csv", 'w')
writer = csv.writer(csvFile,lineterminator="\n")
writer.writerows(row)
csvFile.close()