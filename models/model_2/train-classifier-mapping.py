import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="3";  

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
from  keras import layers, models
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger, ReduceLROnPlateau, TensorBoard
import json
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU, LeakyReLU
#import keras

'''
session_conf = tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)


from keras.backend import tensorflow_backend as K

#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2))
K.set_session(sess)
'''



'''
def train_model(model, video_data_fn="video_data.h5", validation_ratio=0.3, batch_size=32):
    """ Train the video classification model
    """
    with h5py.File(video_data_fn, "r") as video_data:
         sample_count = video_data["train_img"].shape[0]
         sample_idxs = range(0, sample_count)
         sample_idxs = np.random.permutation(sample_idxs)
         training_sample_idxs = sample_idxs[0:int((1-validation_ratio)*sample_count)]
         validation_sample_idxs = sample_idxs[int((1-validation_ratio)*sample_count):]
         training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                   video_data=video_data,
                                                                   training_sample_idxs=training_sample_idxs)
         validation_sequence_generator = generate_validation_sequences(batch_size=batch_size,
                                                                       video_data=video_data,
                                                                       validation_sample_idxs=validation_sample_idxs)
         model.fit_generator(generator=training_sequence_generator,
                             validation_data=validation_sequence_generator,
                             samples_per_epoch=len(training_sample_idxs),
                             nb_val_samples=len(validation_sample_idxs),
                             nb_epoch=100,
                             max_q_size=1,
                             verbose=2,
                             class_weight=None,
                             nb_worker=1)
         
         
'''

'''
#Read The Dataset.h5
fname_in = "dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    train_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    train_Y  = np.array(hf.get('train_Y'), dtype=np.int)

X, Y = train_X1, train_Y
Y = np.reshape(Y, (-1,2))
'''



# define base model
def model_1():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(3 , 100, 100)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
	
    return model


def VGG16():
    
    
    # Determine proper input shape
    input_shape = (128, 128, 200)
    img_input = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    #x = layers.Dense(2, name='predictions')(x)
   

    inputs = img_input
    # Create model.
    #model = models.Model(inputs, x, name='vgg16')

    y1 = layers.Dense(11, activation='softmax')(x)
    y2 = layers.Dense(11, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=[y1, y2])
    
    #model.load_weights("weight/model_1_v1_tf.hdf5")
    
    adam = Adam(lr=0.0001) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model



def model_2(input_shape):
    

    input_size = Input(shape=(input_shape))
    conv1 = Conv2D(16, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(input_size)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(conv1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    merge1 = concatenate([conv1, conv2], axis= 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(merge1)
    
    conv3 = Conv2D(32, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(pool4)
    conv4 = Conv2D(64, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(conv3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    
    conv5 = Conv2D(32, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(pool1)
    conv6 = Conv2D(64, kernel_size=4, activation='relu', strides=(1, 1), padding='same')(conv5)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv6)
    
    merge2 = concatenate([pool2, pool5], axis= 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(merge2)
    
    
    
    flat = Flatten()(pool3)
    hidden1 = Dense(1024, activation='relu')(flat)
    dropout1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(1024, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(hidden2)
    output = Dense(2)(dropout2)
    model = Model(inputs=input_size, outputs=output)
    
    return model


def multi_model():    
    model_input = Input(shape=(128, 128, 200))
    
    x = Conv2D(32, (5, 5),padding='same')(model_input)
    x = LeakyReLU(alpha=0.02)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(196, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
              
    x = GlobalMaxPooling2D()(x)
    
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.02)(x)
    x = Dropout(0.5)(x)
    
    y1 = Dense(11, activation='softmax')(x)
    y2 = Dense(11, activation='softmax')(x)
    
    model = Model(inputs=model_input, outputs=[y1, y2])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


#model.summary()

#history = model.fit(X, Y, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

'''
fname_in = "../data/test/test-dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    test_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    test_Y  = np.array(hf.get('train_Y'), dtype=np.int)


new = np.expand_dims(test_X1[0], axis=0)
ynew=model.predict(new)
print("Predicted=%s" % (ynew[0]))
'''

def save_model_to_json(model, model_name, version, backend):
    if not os.path.exists('model'):
        os.makedirs('model')
    print('saving model...')
    with open('model/' + model_name + '_v'+ version +'_'+ backend +'.json', 'w') as f:
        f.write(model.to_json())
    print('model saved...')    
    
def save_history_json(history, model_name, version, backend):
    with open('history/' + model_name + '_v'+ version +'_'+ backend + '.json', mode='w') as f:
        json.dump(history.history, f)

def model_summary(model):
    
    with open('report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
         
   
def plot(history, model_name, version, backend): 
    if not os.path.exists('graph'):
        os.makedirs('graph')
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')           
    plt.legend(['train', 'validation'], loc='upper left')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig('graph/' + model_name + '_v'+ version +'_'+ backend  +'_accuracy.png', dpi=1000)


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig('graph/' + model_name + '_v'+ version +'_'+ backend  +'_loss.png', dpi=1000)

        
def generate_training_sequences(batch_size, image_data, training_sample_idxs):
    """ Generates training sequences on demand
    """
    while True:
        # generate sequences for training
        training_sample_count = len(training_sample_idxs)
        batches = int(training_sample_count/batch_size)
        remainder_samples = training_sample_count%batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = training_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)

            X = image_data["train_img"][batch_idxs]
            Y = image_data["train_labels"][batch_idxs]
            #Y = np.reshape(Y, (-1,2))
            #print(Y.shape)
            Y = Y.reshape(Y.shape[0], 22)
            Y_train_list = [Y[:, :11], Y[:,11:]]
            
            #yield (np.array(X), np.array(Y))
            yield (np.array(X), Y_train_list)

def generate_validation_sequences(batch_size, image_data, validation_sample_idxs):
    """ Generates validation sequences on demand
    """
    while True:
        # generate sequences for validation
        validation_sample_count = len(validation_sample_idxs)
        batches = int(validation_sample_count/batch_size)
        remainder_samples = validation_sample_count%batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = validation_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = validation_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)

            X = image_data["train_img"][batch_idxs]
            Y = image_data["train_labels"][batch_idxs]
            #Y = np.reshape(Y, (-1,2))
            
            Y = Y.reshape(Y.shape[0], 22)
            Y_train_list = [Y[:, :11], Y[:,11:]]
            
            #yield (np.array(X), np.array(Y))
            yield (np.array(X), Y_train_list)
        



def generate_test_sequences(batch_size, image_data, test_sample_idxs):
    """ Generates validation sequences on demand
    """
    while True:
        # generate sequences for validation
        test_sample_count = len(test_sample_idxs)
        batches = int(test_sample_count/batch_size)
        remainder_samples = test_sample_count%batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = test_sample_idxs[idx*batch_size:]
            else:
                batch_idxs = test_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
            batch_idxs = sorted(batch_idxs)

            X = image_data["test_img"][batch_idxs]
            Y = image_data["test_labels"][batch_idxs]
            #Y = np.reshape(Y, (-1,2))
            
            Y = Y.reshape(Y.shape[0], 22)
            Y_train_list = [Y[:, :11], Y[:,11:]]
            
            #yield (np.array(X), np.array(Y))
            yield (np.array(X), Y_train_list)     
            
def evaluate_model(model, data_fn, batch_size):
    """ Train the video classification model
    """
    with h5py.File(data_fn, "r") as image_data:
         sample_count = image_data["test_img"].shape[0]
         sample_idxs = range(0, sample_count)
         sample_idxs = np.random.permutation(sample_idxs)
         test_sample_idxs = sample_idxs[0:int(sample_count)]
         
         test_sequence_generator = generate_test_sequences(batch_size=batch_size,
                                                                   image_data=image_data,
                                                                   test_sample_idxs=test_sample_idxs)
         
         loss, dense_2_loss, dense_3_loss, dense_2_acc, dense_3_acc = model.evaluate_generator(generator=test_sequence_generator,
                             steps = len(test_sample_idxs)/batch_size,
                             verbose=1)
         return loss, dense_2_loss, dense_3_loss, dense_2_acc, dense_3_acc
     
        
            
def predict_model(model, data_fn, batch_size):
    """ Train the video classification model
    """
    with h5py.File(data_fn, "r") as image_data:
         sample_count = image_data["test_img"].shape[0]
         sample_idxs = range(0, sample_count)
         sample_idxs = np.random.permutation(sample_idxs)
         test_sample_idxs = sample_idxs[0:int(sample_count)]
         
         test_sequence_generator = generate_test_sequences(batch_size=batch_size,
                                                                   image_data=image_data,
                                                                   test_sample_idxs=test_sample_idxs)
         
         predictions = model.predict_generator(generator=test_sequence_generator,
                             steps = len(test_sample_idxs)/batch_size,
                             verbose=1)
         return predictions
     
def train_model(model, data_fn, validation_ratio, batch_size, model_name, version, backend):
    """ Train the video classification model
    """
    with h5py.File(data_fn, "r") as image_data:
         sample_count = image_data["train_img"].shape[0]
         sample_idxs = range(0, sample_count)
         sample_idxs = np.random.permutation(sample_idxs)
         
         training_sample_idxs = sample_idxs[0:int(sample_count)]
         
         #training_sample_idxs = sample_idxs[0:int((1-validation_ratio)*sample_count)]
         #validation_sample_idxs = sample_idxs[int((1-validation_ratio)*sample_count):]
         training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                   image_data=image_data,
                                                                   training_sample_idxs=training_sample_idxs)
         #validation_sequence_generator = generate_validation_sequences(batch_size=batch_size,
         #                                                             image_data=image_data,
         #                                                             validation_sample_idxs=validation_sample_idxs)

         
         
         if not os.path.exists('weight'):
             os.makedirs('weight')
             
         tl_filepath='weight/' + model_name + '_v' + version +'_'+ backend + '.hdf5'
         checkpoint = ModelCheckpoint(tl_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        
         early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')
        
         reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1, min_lr=1e-20) 
        
         if not os.path.exists('history'):
            os.makedirs('history')
         csv_logger = CSVLogger('history/' + model_name + '_v' + version +'_'+ backend + '.log', append=True)
          
         callbacks_list = [checkpoint, csv_logger] 

         history = model.fit_generator(generator=training_sequence_generator,
                             #validation_data=validation_sequence_generator,
                             steps_per_epoch = len(training_sample_idxs)/batch_size,
                             #validation_steps=len(validation_sample_idxs)/batch_size,
                             epochs=100,
                             verbose=1,
                             callbacks=callbacks_list)
         return history
     
        
        
     
if __name__ == '__main__':        
        
    model_name = 'model_2'
    version = '1'
    backend = 'tf'
    validation_ratio=0.1
    batch_size=16
    data_fn="../../dataset_mapping.hdf5"
    img_rows, img_cols, channel = 128, 128, 200
    input_shape = (img_rows, img_cols, channel)
    #model = model_1()
    model = VGG16()
    
    #model = multi_model()
    #model.summary()
    
    #save_model_to_json(model, model_name, version, backend)
    
    #model_summary(model)
    
    # Compile model
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    
    history = train_model(model, data_fn, validation_ratio, batch_size, model_name, version, backend)
    #model_summary(model)
    
    #plot(history, model_name, version, backend)
    save_history_json(history, model_name, version, backend)
    
    #loss, acc = evaluate_model(model, data_fn, batch_size)
    
    
    
    
    #print('acc:', acc)
    #print('loss:', loss)
    
    
    #multi_score = multi_model.evaluate(x_test3, y_test3_list, verbose=0)
    
    loss, dense_2_loss, dense_3_loss, dense_2_acc, dense_3_acc = evaluate_model(model, data_fn, batch_size)


    
    #print("Scores: \n" , (multi_score))
    print("First label. Accuracy: %.2f%%" % (dense_3_acc*100))
    print("Second label. Accuracy: %.2f%%" % (dense_3_acc*100))


    #predictions = predict_model(model, data_fn, batch_size)
    
    #print("Scores: \n" , (multi_score))
    #print("First label. Accuracy: %.2f%%" % (multi_score[3]*100))
    #print("Second label. Accuracy: %.2f%%" % (multi_score[4]*100))
