from random import shuffle
import cv2
import numpy as np
import h5py
import os
import tensorflow as tf
import time
import csv
from datetime import timedelta
import matplotlib

import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Read The Dataset.h5
fname_in = "../data/dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    train_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    train_Y  = np.array(hf.get('train_Y'), dtype=np.int)

X, Y = train_X1, train_Y
Y = np.reshape(Y, (-1,2))

img_rows, img_cols, channel = 128, 128, 200
input_shape = (img_rows, img_cols, channel)
# define base model
def model_1():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 200)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    return model


model = model_1()
model.summary()
history = model.fit(X, Y, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)


fname_in = "../data/test/test-dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    test_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    test_Y  = np.array(hf.get('train_Y'), dtype=np.int)


new = np.expand_dims(test_X1[0], axis=0)
ynew=model.predict(new)
print("Predicted=%s" % (ynew[0]))