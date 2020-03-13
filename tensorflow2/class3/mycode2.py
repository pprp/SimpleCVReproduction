# coding=utf-8
# tf2.0 mnist fit

import tensorflow as tf
import os

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

num_classes = 10
total_eopch = 20

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(3)

loss_sce = tf.keras.losses.sparse_categorical_crossentropy
optimizer = tf.keras.optimizers.SGD(0.01)

inputs = Input(shape=(784, ))
x = Dense(128, activation="relu")(inputs)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# 记录指标
train_loss = tf.keras.metrics.Mean(name='trainloss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='trainacc')
test_loss = tf.keras.metrics.Mean(name='testloss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='testacc')

# 回调函数
checkpoint_path = "./checkpoints"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 verbose=1,
                                                 period=1)

model.compile(optimizer=optimizer, loss=loss_sce, metrics=[train_acc])

model.fit(train_ds, epochs=total_eopch)

loss, acc = model.evaluate(train_ds)

print("saved model, loss: {:5.2f}, acc: {:5.2f}".format(loss, acc))
