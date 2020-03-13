import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                                     concatenate)
from tensorflow.keras.utils import plot_model

num_classes = 10
total_epoch = 30
mnist = tf.keras.datasets.mnist

#1. prepare datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 2, -1)
x_test = x_test.reshape(x_test.shape[0], 2, -1)

y_train = tf.one_hot(y_train, num_classes)
y_test = tf.one_hot(y_test, num_classes)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#2. net_build
inputs = Input(shape=(392, ), name="D1_input")
outputs = Dense(num_classes, name="D1")(inputs)
share_base = Model(inputs=inputs, outputs=outputs, name="seq1")

x1 = Input(shape=(392, ), name="input_1")
x2 = Input(shape=(392, ), name="input_2")
s1 = share_base(x1)
s2 = share_base(x2)

b = K.zeros(shape=(10))
x = s1 + s2 + b
x = Activation('softmax', name='activation')(x)

siamese_net = Model(inputs=[x1, x2], outputs=x)

plot_model(siamese_net,
           to_file='./siamese_net.png',
           show_shapes=True,
           expand_nested=True)

#3. train and test
loss_ce = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(3e-4)

# metrics用于记录指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_loss')


@tf.function
def train_step(images, labels):
    part1 = images[:, 0]
    part2 = images[:, 1]
    with tf.GradientTape() as tape:
        outputs = siamese_net([part1, part2])
        loss = loss_ce(labels, outputs)
    gradients = tape.gradient(loss, siamese_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, siamese_net.trainable_variables))

    train_loss(loss)
    train_acc(labels, outputs)


@tf.function
def test_step(images, labels):
    part1 = images[:, 0]
    part2 = images[:, 1]
    outputs = siamese_net([part1, part2])
    loss = loss_ce(labels, outputs)

    test_loss(loss)
    test_acc(labels, outputs)


for epoch in range(total_epoch):
    train_acc.reset_states()
    train_loss.reset_states()
    test_acc.reset_states()
    test_loss.reset_states()
    for images, labels in train_ds:
        train_step(images, labels)

    for images, labels in test_ds:
        test_step(images, labels)

    print(
        "epoch:%d,train loss:%.3f,train acc:%.3f,test loss:%.3f,test acc:%.3f"
        % (epoch, train_loss.result(), train_acc.result() * 100,
           test_loss.result(), test_acc.result() * 100))

#4. draw weights of 10 classes
train_weights=siamese_net.get_layer('seq1').get_layer('D1').kernel.numpy()

num = np.arange(0, 392, 1, dtype="float")
num = num.reshape((14, 28))
plt.figure(num='Weights', figsize=(10, 10))  # 创建一个名为Weights的窗口,并设置大小
for i in range(10):  # W.shape[1]
    num = train_weights[:, i: i+1].reshape((14, -1))
    plt.subplot(2, 5, i + 1)
    num = num * 255.
    plt.imshow(num, cmap=plt.get_cmap('hot'))
    plt.title('weight %d image.' % (i + 1))  # 第i + 1幅图片
plt.show()
print(np.min(num))
print(np.max(num))
