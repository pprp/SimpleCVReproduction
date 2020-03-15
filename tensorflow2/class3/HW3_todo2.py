import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten

tf.keras.backend.set_floatx('float64')
mnist = keras.datasets.mnist

######################
EPOCHS = 100
BATCH_SIZE = 1000
LEN_IMAGE_SIZE = 784

lr = 3e-4
e_w = 1.0
iters = 5
######################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)


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

    idx_same = np.where(y1 == y2) # 找到相同的下角标
    idx_rand = np.random.randint(BATCH_SIZE, size=len(idx_same))  # 随机取样
    index = np.union1d(idx_same, idx_rand).astype(np.int64)  # 所有需要取样的样本

    data_list = []
    label_list = []

    judge = np.array(y1 != y2)

    for ix in index:
        data_list.append([x1[ix], x2[ix]])
        label_list.append(judge[ix])

    return np.array(data_list), np.array(label_list)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(500, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        embedding = self.d2(x)
        return embedding


def dist(output1, output2):
    E = K.sqrt(K.sum(K.square(output1 - output2), 1))  # dim=1
    return E


def loss_object(Y, E, Q=1):
    pos_loss = Y * 2 * Q * K.exp((-2.77 * E) / Q)
    neg_loss = 2 * (1 - Y) * (E**2) / Q
    return pos_loss + neg_loss


model = MyModel()

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

            # print(label.shape, E.shape)

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
