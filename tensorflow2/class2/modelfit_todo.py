
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow_core.python.keras import Sequential,Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D
import time
#if you have omp problem in mac
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#from tensorflow.python.framework import graph_util

# data prepare 
x_t = np.arange( ,  ,  ,dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = np.cos(  * x_t +  )

# define model
inputs = tf.keras.Input()
outputs = tf.keras.layers.Dense(units=, input_dim=)(inputs)
Lm1 = tf.keras.Model(inputs=, outputs=, name=)

checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(,save_weights_only=True,verbose=1, period=200) 

Lm1.(optimizer=,loss=, metrics=)
Lm1.(, , epochs=, )
loss,acc= Lm1.evaluate(x_train, y_train)
print("saved model, loss: {:5.2f}".format(loss))


forecast=Lm1(x_train)
plt.figure()
plot1 = plt.plot(, , 'b', label='original values')
plot2 = plt.plot(, , 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()

latest = 
Lm2 = tf.keras.Model(inputs=, outputs=, name=)
Lm2.compile(loss='mse')
Lm2.load_weights(latest)
loss= Lm2.evaluate(x_train, y_train)
print("Restored model, loss: {:5.2f}".format(loss))

forecast=Lm2(x_train)
plt.figure()
plot1 = plt.plot(, , 'b', label='original values')
plot2 = plt.plot(, , 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()