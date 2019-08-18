import glob
import re
import h5py
import numpy as np
import cv2
from datetime import datetime
import csv
import scipy.io
from random import shuffle
from keras.utils.np_utils import to_categorical   


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def mapping(x):
    if x == -1:
        return 6
    elif x == -2:
        return 7
    elif x == -3:
        return 8
    elif x == -4:
        return 9
    elif x == -5:
        return 10
    else:
        return x

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'dataset_mapping.hdf5'  # address to where you want to save the hdf5 file
train_parent_directory = sorted(glob.glob('data/*.mat'), key=numericalSort)
# read addresses and labels from the 'train' folder
addrs = train_parent_directory
#labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
'''
if shuffle_data:
    c = list(zip(addrs))
    shuffle(c)
    addrs, labels = zip(*c)
'''


# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.9*len(addrs))]
train_labels = addrs[0:int(0.9*len(addrs))]

test_addrs = addrs[int(0.9*len(addrs)):]
test_labels = addrs[int(0.9*len(addrs)):]

train_shape = (len(train_addrs), 128, 128, 200)

test_shape = (len(test_addrs), 128, 128, 200)
train_Y_shape = (len(train_labels), 2, 11)
test_Y_shape = (len(test_labels), 2, 11)
    
    
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.float32)

hdf5_file.create_dataset("test_img", test_shape, np.float32)
#hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)




hdf5_file.create_dataset("train_labels", train_Y_shape, np.int8)

hdf5_file.create_dataset("test_labels", test_Y_shape, np.int8)




# a numpy array to save the mean of the images
#mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]

    mat = scipy.io.loadmat(addr)
    
    img = mat["arr"]
    vector = mat["vector"]   
    
    new_vector = vector
    new_vector[0][0] = mapping(vector[0][0])
    new_vector[0][1] = mapping(new_vector[0][1])
    
    

    categorical_labels = to_categorical(new_vector[0], num_classes=11)
    
    print(addr)

    # save the image and calculate the mean so far
    
    hdf5_file["train_img"][i, ...] = img[None]
    hdf5_file["train_labels"][i, ...] = categorical_labels[None]
    
    
    #mean += img / float(len(train_labels))

print('------------------')
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    
    mat = scipy.io.loadmat(addr)
    
    img = mat["arr"]
    vector = mat["vector"]
    
    new_vector = vector
    new_vector[0][0] = mapping(vector[0][0])
    new_vector[0][1] = mapping(new_vector[0][1])
    
    categorical_labels = to_categorical(new_vector[0], num_classes=11)
    
    
    print(addr)

    # save the image and calculate the mean so far
    
    hdf5_file["test_img"][i, ...] = img[None]
    hdf5_file["test_labels"][i, ...] = categorical_labels[None]
    
# save the mean and close the hdf5 file
#hdf5_file["train_mean"][...] = mean
hdf5_file.close()
