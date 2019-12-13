from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model('./my_model/5_model2.h5')
# model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()