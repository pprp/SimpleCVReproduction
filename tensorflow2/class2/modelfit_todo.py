import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow_core.python.keras import Sequential, Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D
import time
#if you have omp problem in mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#from tensorflow.python.framework import graph_util
a = 20
b = 14

# data prepare
x_t = np.arange(-1 * b / a, (2 * np.pi - b) / a,
                np.pi / (a * 1000),
                dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(a * x_t + b)

# define model
inputs = tf.keras.Input(shape=(3, ), name="inputs")
outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
Lm1 = tf.keras.Model(inputs=inputs, outputs=outputs, name="fit")

Lm1.summary()

checkpoint_path = "./checkpoints/weights.{epoch:02d}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor="mse",
                                                 mode='min',
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

Lm1.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            loss='mse',
            metrics=['mse'])

Lm1.fit(x_train, y_train, epochs=1000, batch_size=64, callbacks=[cp_callback])
loss, acc = Lm1.evaluate(x_train, y_train)
print("saved model, loss: {:5.2f}".format(loss))

forecast = Lm1(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plot2 = plt.plot(x_t, forecast, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()

latest = "./checkpoints/weights.1000.h5"
Lm2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="fit")
Lm2.compile(loss='mse')
Lm2.load_weights(latest)
loss = Lm2.evaluate(x_train, y_train)
print("Restored model, loss: {:5.2f}".format(loss))

forecast = Lm2(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plot2 = plt.plot(x_t, forecast, 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()