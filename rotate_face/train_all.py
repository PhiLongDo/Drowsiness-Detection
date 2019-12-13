from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_model(window_size):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(window_size, 10)),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),      
        keras.layers.Dense(3, activation=tf.nn.softmax)
        ])
    #Model I
    model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

for window_size in [5]:              
        print ('*************************************************\nCreate model')
        Blinks = np.load('./data_preprocess/Blinks_'+str(window_size)+'_Fold5.npy')
        Labels = np.load('./data_preprocess/Labels_'+str(window_size)+'_Fold5.npy')
        BlinksTest = np.load('./data_preprocess/BlinksTest_'+str(window_size)+'_Fold5.npy')
        LabelsTest = np.load('./data_preprocess/LabelsTest_'+str(window_size)+'_Fold5.npy')

        Blinks = np.concatenate((Blinks,BlinksTest),axis=0)
        Labels = np.concatenate((Labels,LabelsTest),axis=0)

        model = create_model(window_size)

        print('-----------------Training-------------------')
        # model.fit(Blinks, np.squeeze(np.asarray(Labels)), epochs=80)
        model.fit(Blinks, Labels, batch_size=64, epochs=80)

        print ('Saving...')
        model.save('./my_model/'+str(window_size)+'_model_all.h5')
        print('\n----------------------------------------------------------------------------------------------\n\n')

