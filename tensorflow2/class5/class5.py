
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense,
                                     MaxPooling2D, Conv2D,
                                     Flatten, multiply)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


np.random.seed(0)
np.set_printoptions(precision=4)

tf.random.set_seed(0)

#################### 0. 参数 ####################

BATCH_SIZE = 100
EPOCHS = 10

#################### 1. 数据 ####################

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print(x_train.shape)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

#################### 2. SELeNet ####################


def SeNetBlock(feature, reduction=4):
    temp = feature
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channels = temp.shape[channel_axis]
    # 得到feature的通道数量w
    avg_x = tf.keras.layers.GlobalAveragePooling2D()(temp)
    # 先对feature的每个通道进行全局平均池化Global Average Pooling 得到通道描述子（Squeeze）
    x = tf.keras.layers.Dense(
        units=int(channels)//reduction, activation=tf.nn.relu, use_bias=False)(avg_x)
    # 接着做reduction，用int(channels)//reduction个卷积核对 avg_x做1x1的点卷积
    x = tf.keras.layers.Dense(units=int(channels), use_bias=False)(x)
    # 接着用int(channels)个卷积核个数对 x做1x1的点卷积，扩展x回到原来的通道个数
    se_feature = tf.keras.activations.sigmoid(x)  # 对x 做 sigmoid 激活
    return multiply([se_feature, feature]), se_feature
    # 返回以cbam_feature 为scale，对feature做拉伸加权的结果（Excitation）


input = Input(shape=(28, 28, 1))
x = Conv2D(6, (5, 5), (1, 1), activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Conv2D(16, (5, 5), (1, 1),
           activation='relu', padding='valid')(x)
# SE Module
x, se_feature = SeNetBlock(x)
x = MaxPooling2D(2, strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(input, [x, se_feature], name="SELeNet")

model.summary()

plot_model(model, to_file="SELeNet.png")

#################### 3. 工具集 ####################

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

#################### 4. 训练&测试 ####################


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions, se_feature = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions, se_feature = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    return se_feature

#################### 5. train ####################

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        se_feature = test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

#################### 6. test ####################


for i, (images, labels) in enumerate(test_ds):
    se_feature = test_step(images, labels)
    
    labels = tf.cast(labels, tf.int32)
    sorted = tf.argsort(labels)
    labels = tf.gather(labels, sorted)
    se_feature = tf.gather(se_feature, sorted)

    se_feature_numpy = se_feature.numpy()

    plt.figure()
    img = plt.imshow(se_feature_numpy)
    img.set_cmap('hot')
    plt.savefig("./%d_se_feature.png" % i)
    plt.close()