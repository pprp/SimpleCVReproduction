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

train_dataset = 

#try "tf.keras.losses.MeanSquaredError" without "()" as well?
loss_object = tf.keras.losses.MeanSquaredError()
test_loss = tf.keras.metrics.Mean(name='test_loss')

#load model here
model = 

def test_step(data, labels):
  predictions=
  t_loss = 

  test_loss(t_loss)

EPOCHS = 5

for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  test_loss.reset_states()

  for ,  in :
    test_step(, )

  template = 'Epoch {},  Test Loss: {}'
  print (template.format(epoch+1,
                         test_loss.result()))

model.compile(loss='mse')
loss= model.evaluate(x_train, y_train)
print("Restored model, loss: {:5.2f}".format(loss))


forecast=model(x_train)
plt.figure()
plot1 = plt.plot(, , 'b', label='original values')
plot2 = plt.plot(, , 'r', label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.show()