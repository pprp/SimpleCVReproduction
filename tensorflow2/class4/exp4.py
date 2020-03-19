import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')
mnist = keras.datasets.mnist

######################
EPOCHS = 100
BATCH_SIZE = 1000

lr = 3e-4
e_w = 1.0
iters = 5
######################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)


class TFLeNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(TFLeNet, self).__init__(name='TFLeNet')
        self.num_classes = num_classes
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(
            5, 5), strides=(1, 1), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(
            5, 5), strides=(1, 1), activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

# 查看mnist 图片


def minist_draw(im):
    im = im.reshape(28, 28)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.savefig("test.png")
    plt.close()


def balanced_batch(batch_x, batch_y, num_cls):
    # batch_x MNIST样本 batch_y, MNIST标签 num_cls 
    # （数字类型个数，10，为了让10个数字类型都充分采样正负样本对）
    batch_size = len(batch_y)
    
    pos_per_cls_e = round(batch_size/2/num_cls/2)  # bs最少40+
    pos_per_cls_e *= 2

    # 根据y进行排序
    index = batch_y.argsort()
    ys_1 = batch_y[index]

    num_class = []
    pos_samples = []
    neg_samples = set()

    cur_ind = 0

    for item in set(ys_1):
        num_class.append((ys_1 == item).sum()) 
        # 记录有多少个一样的
        num_pos = pos_per_cls_e

        while(num_pos > num_class[-1]):
            num_pos -= 2
        # 找一个恰好大于num_class个数的值
        # 作为选取的正样本的个数

        pos_samples.extend(np.random.choice(
            index[cur_ind:cur_ind+num_class[-1]],
            num_pos, replace=False).tolist())
        # 正样本

        neg_samples = neg_samples | (set(index[cur_ind:cur_ind+num_class[-1]])-set(list(pos_samples)))
        cur_ind += num_class[-1]

    neg_samples = list(neg_samples)
    
    x1_index = pos_samples[::2]
    x2_index = pos_samples[1:len(pos_samples)+1:2]

    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples)+1:2])
    
    p_index = np.random.permutation(len(x1_index))  # shuffle操作
    
    x1_index = np.array(x1_index)[p_index]
    x2_index = np.array(x2_index)[p_index]
    
    r_x1_batch = batch_x[x1_index]
    r_x2_batch = batch_x[x2_index]
    # 得到最终重排后的结果

    r_y_batch = np.array(batch_y[x1_index] !=
                         batch_y[x2_index], dtype=np.float32)

    return r_x1_batch, r_x2_batch, r_y_batch


def balance_sample(train_ds, test_ds, train=True):
    train_ds = iter(train_ds)
    test_ds = iter(test_ds)
    if train:
        x1, y1 = next(train_ds)
        x2, y2 = next(train_ds)
    else:
        x1, y1 = next(test_ds)
        x2, y2 = next(test_ds)

    y1 = y1[..., np.newaxis]
    y2 = y2[..., np.newaxis]

    idx_same = np.where(y1 == y2)  # 找到相同的下角标
    idx_rand = np.random.randint(BATCH_SIZE, size=len(idx_same))  # 随机取样
    index = np.union1d(idx_same, idx_rand).astype(np.int64)  # 所有需要取样的样本

    data_list = []
    label_list = []

    judge = np.array(y1 != y2)

    for ix in index:
        data_list.append([x1[ix], x2[ix]])
        label_list.append(judge[ix])

    return np.array(data_list), np.array(label_list)


def dist(output1, output2):
    E = K.sqrt(K.sum(K.square(output1 - output2), 1))  # dim=1
    return E


def loss_object(Y, E, Q=1):
    pos_loss = 2 * (1 - Y) * (E**2) / Q
    neg_loss = Y * 2 * Q * K.exp((-2.77 * E) / Q)
    return pos_loss + neg_loss


def loss(label_pair, e_w, Q=1):
    loss_p = (1 - label_pair) * 2 / Q * e_w ** 2
    loss_n = label_pair * 2 * Q * K.exp(-2.77 / Q * e_w)
    loss = K.mean(loss_p + loss_n)
    return loss


model = TFLeNet()

optimizer = tf.keras.optimizers.Adam(lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

# @tf.function


def train_epoch(train_ds):
    for i in range(iters):
        with tf.GradientTape() as tape:
            data, label = balance_sample(train_ds, test_ds, train=True)
            output1 = model(data[:, 0])
            output2 = model(data[:, 1])
            E = dist(output1, output2)

            label = np.squeeze(label, 1)

            loss = loss_object(label, E)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        E = K.cast(E >= e_w, dtype='float64')
        train_loss(loss)
        train_accuracy(label, E)


# @tf.function
def test_epoch(test_ds):
    for i in range(iters):
        data, label = balance_sample(train_ds, test_ds, train=False)
        output1 = model(data[:, 0])
        output2 = model(data[:, 1])
        E = dist(output1, output2)
        loss = loss_object(label, E)
        E = K.cast(E >= e_w, dtype='float64')
        test_loss(loss)
        test_accuracy(label, E)


for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    train_epoch(train_ds)
    test_epoch(test_ds)

    print('Epoch {}, Train Loss: {:.3f}, Train acc: {:.3f} Test Loss: {:.3f} Test acc: {:.3f}'.
          format(epoch + 1, train_loss.result(), train_accuracy.result(),
                 test_loss.result(), test_accuracy.result()))
