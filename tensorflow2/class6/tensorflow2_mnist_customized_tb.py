import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import io
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

n_classes = 10
#y_train = tf.keras.utils.to_categorical(y_train, n_classes)
#y_test = tf.keras.utils.to_categorical(y_test, n_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
class MyLeNet(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyLeNet, self).__init__(name='myLeNet')
        self.num_classes = num_classes
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')
        self.pool1= tf.keras.layers.MaxPooling2D(2,strides=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')
        self.pool2 = tf.keras.layers.MaxPooling2D(2,strides=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='relu')
        self.dense2 = tf.keras.layers.Dense(84, activation='relu')
        self.dense3 =tf.keras.layers.Dense(num_classes, activation='softmax')
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


model=MyLeNet()


def plot_to_image(figure):
    buf = io.BytesIO()  # 在内存中存储画
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # 传化为TF 图
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images):
    # 返回一个5x5的mnist图像
    figure  = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure

#plot_model(model,'tett.png',expand_nested=True)

# 创建监控类，监控数据写入到log_dir目录
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('summary_logs','gradient_tape',current_time,'train')
test_log_dir = os.path.join('summary_logs','gradient_tape',current_time,'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


#try SparseCategoricalCrossentropy without one-hot
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.SGD(0.01)

#why not accuracy?
train_loss = tf.keras.metrics.Mean(name='train_loss')
#try SparseCategoricalAccuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#try not use tf.function to debug
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
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
last_step=0
tlast_step=0
for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()


  for step,(images, labels) in enumerate(train_ds):
    if step+last_step==0:
        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        tf.summary.trace_on(graph=True, profiler=True)
        # Call only one tf.function when tracing.
        train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="lenet_train_trace",
                step=step,
                profiler_outdir=train_log_dir)
    else:
        train_step(images, labels)
        with train_summary_writer.as_default():
          tf.summary.scalar('loss', train_loss.result(), step=step+last_step)
          tf.summary.scalar('accuracy', train_accuracy.result(), step=step+last_step)   

          if step%500==0:
              val_images = images[:25]
              val_images = tf.reshape(val_images, [-1, 28, 28, 1])

              tf.summary.image("val-onebyone-images:", val_images, max_outputs=25,step=step+last_step)  # 可视化测试用图片，25张
              val_images = tf.reshape(val_images, [-1, 28, 28])
              figure = image_grid(val_images)
              tf.summary.image('val-images:', plot_to_image(figure),step=step+last_step)
  
  last_step=step+last_step
  for step,(test_images, test_labels) in enumerate(test_ds):
    test_step(test_images, test_labels)
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=last_step)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=last_step)


  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
  #model.summary()
  #plot_model(model,'tett.png',expand_nested=True)