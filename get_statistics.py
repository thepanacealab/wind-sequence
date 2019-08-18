
from random import shuffle

import numpy as np
import h5py
import os

import time
import csv
from datetime import timedelta


import json

#import keras

   
def save_stat(x, save_dir, filename):
    data = [[x[10], x[9], x[8], x[7], x[6], x[0], x[1], x[2], x[3], x[4], x[5]]]
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    file_name = str(filename)+'.csv'
    
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4','5'])
        writer.writerows(data)
    
    csvFile.close()         


data_fn="dataset_mapping.hdf5"


x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with h5py.File(data_fn, "r") as image_data:
    for i in image_data['train_labels']:
        vector_x = i[0]
        vector_y = i[1]
        
        
        vector_class_x = np.argmax(vector_x, axis=-1)
        vector_class_y = np.argmax(vector_y, axis=-1)
    
        x[vector_class_x] = x[vector_class_x] +1
        y[vector_class_y] = y[vector_class_y] +1    
        
        
        total[vector_class_x] = total[vector_class_x] +1
        total[vector_class_y] = total[vector_class_y] +1    
        
        

        #break

save_dir = 'csv'

filename = 'x'

save_stat(x, save_dir, filename)

filename = 'y'
save_stat(y, save_dir, filename)


filename = 'total'
save_stat(total, save_dir, filename)

