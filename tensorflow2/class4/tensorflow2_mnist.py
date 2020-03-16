import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)
x = tf.keras.layers.MaxPooling2D(2,strides=(2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(84, activation='relu')(x)
outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)


model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')

model.summary()

model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs\\", histogram_freq=1)

model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
