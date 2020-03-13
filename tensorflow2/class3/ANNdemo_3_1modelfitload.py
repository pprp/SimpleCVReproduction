
# coding=utf-8
# TF2.0 应用model fit训练mnist的简单案例

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow_core.python.keras import Sequential,Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D,Input

import os

import numpy as np

num_classes=10
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.one_hot(y_train, num_classes)
y_test = tf.one_hot(y_test, num_classes)
x_train=x_train.reshape((60000,-1))
x_test=x_test.reshape((10000,-1))
# 添加一个通道维度
#x_train = x_train[..., tf.newaxis]
#x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(64)
#print(train_ds)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


checkpoint_path = "./checkpoints/saved_model.pb"
checkpoint_dir = os.path.dirname(checkpoint_path)


Lm2 = tf.keras.models.load_model(checkpoint_dir)
Lm2.compile(loss='categorical_crossentropy',metrics=['CategoricalAccuracy'])
loss,acc= Lm2.evaluate(train_ds)
print("saved model, loss: {:5.2f}, acc: {:5.2f}".format(loss,acc))



