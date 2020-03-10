# coding=utf-8
# TF2.0 应用model fit训练mnist的简单案例,官方案例

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow_core.python.keras import Sequential,Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D

import numpy as np

num_classes=10
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# y_train = tf.one_hot(y_train, num_classes)
# y_test = tf.one_hot(y_test, num_classes)
x_train=x_train.reshape((60000,-1))
x_test=x_test.reshape((10000,-1))
# 添加一个通道维度
#x_train = x_train[..., tf.newaxis]
#x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(64)
#print(train_ds)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


model=Sequential()
model.add(Dense(num_classes,activation='softmax'))

#try SparseCategoricalCrossentropy without one-hot
loss_object = tf.keras.losses.categorical_crossentropy

optimizer = tf.keras.optimizers.SGD(0.01)

#why not accuracy?
train_loss = tf.keras.metrics.Mean(name='train_loss')
#try SparseCategoricalAccuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

#try not use tf.function to debug
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        #print(predictions)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))
