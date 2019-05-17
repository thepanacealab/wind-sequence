import glob
import re
import h5py
import numpy as np
import cv2
from datetime import datetime
import csv
import scipy.io

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_x1_img_path_list = []
train_y_img_path_list = []


train_parent_directory = sorted(glob.glob('../data/*.mat'), key=numericalSort)

#TRAIN DATASET
for i in train_parent_directory:
    
    mat = scipy.io.loadmat(i)
    
    data = mat["arr"]
    vector = mat["vector"]        
    train_x1_img_path_list.extend([data])
    train_y_img_path_list.extend(vector)
        
  
        
       
train_X1 = train_x1_img_path_list
train_Y = train_y_img_path_list
   
        
train_X1_shape = (len(train_X1), 128, 128, 200)

train_Y_shape = (len(train_Y), 1, 2)


hdf5_path = '../data/dataset.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')

#hdf5_file.create_dataset("train_X1", train_X1_shape, dtype="f", compression="gzip", compression_opts=4)

hdf5_file.create_dataset("train_X1", train_X1_shape, dtype="f")
hdf5_file.create_dataset("train_Y", train_Y_shape, dtype="i")



# loop over train_x1 addresses
for i in range(len(train_X1)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_X1)))
    img = train_X1[i]
    vector = train_Y[i]

    hdf5_file["train_X1"][i, ...] = img[None]
    hdf5_file["train_Y"][i, ...] = vector[None]
    

hdf5_file.close()

'''
fname_in = "../data/dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    d_train_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    d_train_Y = np.array(hf.get('train_Y'), dtype=np.int)
    
'''