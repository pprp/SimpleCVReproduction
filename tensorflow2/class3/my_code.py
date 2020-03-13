# coding=utf-8
# tf2 运行mnist

import numpy as numpy
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# 参数设置
num_classes = 10
total_epoch = 10

# 数据加载
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize
'''
print(x_train.shape, x_test.shape)
out: (60000, 28, 28) (10000, 28, 28)
type: numpy.ndarray
'''

# 将图片reshape
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# 将y处理为one_hot向量
y_train = tf.one_hot(y_train, num_classes)
y_test = tf.one_hot(y_test, num_classes)

# 构建数据集生成器
# shuffle代表随机数据，batch设置为32
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))\
           .shuffle(1000)\
           .batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 构建模型,用两层全连接层
model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 选择损失函数,注意没有括号
loss_ce = tf.keras.losses.categorical_crossentropy

# 优化器
optimizer = tf.keras.optimizers.SGD(lr=0.01)

# metrics用于记录指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_loss')


# 定义训练和测试
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        outputs = model(images)
        loss = loss_ce(labels, outputs)  # 注意顺序
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # 更新参数
    train_loss(loss)
    train_acc(labels, outputs)


@tf.function
def test_step(images, labels):
    outputs = model(images)
    loss = loss_ce(labels, outputs)
    # 更新参数
    test_loss(loss)
    test_acc(labels, outputs)


# 整体循环
for i in range(total_epoch):
    # 重置
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
        % (i, train_loss.result(), train_acc.result() * 100, test_loss.result(),
           test_acc.result() * 100))
