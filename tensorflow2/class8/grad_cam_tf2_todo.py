from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.layers import Lambda,Multiply
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras
import matplotlib.pyplot as plt
import sys
import cv2

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    #img_path = sys.argv[1]
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
@tf.custom_gradient
def gg_relu(x):
    result = tf.keras.activations.relu(x)
    def guided_grad(grad):
        dtype = x.dtype
        return grad * tf.cast(grad > 0., dtype) * \
            tf.cast(x > 0., dtype)
    return result, guided_grad
class GGReLuLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GGReLuLayer, self).__init__()

    def call(self, x):
        return gg_relu(x)  # you don't need to explicitly define the custom gradient
                             # as long as you registered it with the previous method

def get_class_saliency(model,image,category_index):

    tf_image=K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        predictions =model(tf_image)
        loss = predictions[:, category_index]
    #conv_layer= [l for l in input_model.layers if l.name is layer_name][0]
    saliency = tape.gradient(loss,tf_image)[0]
    saliency =normalize(saliency)
    return saliency

def get_saliency(model,image,layer_name):
    grad_model = tf.keras.models.Model(model.inputs, model.get_layer(layer_name).output)

    tf_image=K.cast(image, dtype='float32')
    with tf.GradientTape() as tape:
        tape.watch(tf_image)
        conv_outputs =grad_model(tf_image)
        max_output = K.sum(K.max(conv_outputs, axis=3))
    #conv_layer= [l for l in input_model.layers if l.name is layer_name][0]
    saliency = tape.gradient(max_output,tf_image)[0]
    saliency =normalize(saliency)
    return saliency

#TODO3: complete the replace fun by repalcing the acivations
def replace_activation_layer_in_keras(model, replaced_layer, activation_fun):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if hasattr(layers[i], 'activation'):
            if layers[i].activation == replaced_layer:
                layers[i].activation = 
        x = 

    new_model = tf.keras.Model(inputs=model.inputs, outputs=x)
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first': #if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000
    #TODO1: make a grad_model with the layer_name conv layer as a output
    grad_model = 
    with tf.GradientTape() as tape:
        conv_outputs, predictions =grad_model(image)   
        loss = predictions[:, category_index]
    #conv_layer= [l for l in input_model.layers if l.name is layer_name][0]
    grads = tape.gradient(loss,conv_outputs)
    grads_val = normalize(grads[0])

    output= conv_outputs[0]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    #TODO2: using the w weights to sum the channels of the conv layers
    for i, w in enumerate(weights):
        cam +=

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

#preprocessed_input = load_image(sys.argv[1])
preprocessed_input = load_image("cat_dog.png")

model = VGG16(weights=None)
model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

predictions = model.predict(preprocessed_input)
top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))


top_k=5
top_k_idx=np.argsort(predictions[0])[::-1][0:top_k]

predicted_class = top_k_idx[2] #np.argmax(predictions)
predicted_tensor=K.one_hot([predicted_class], 1000)


cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
cv2.imwrite("gradcam.jpg", cam)

guided_model = replace_activation_layer_in_keras(model, tf.keras.activations.relu, GGReLuLayer)

saliency = get_saliency(guided_model,preprocessed_input,"block5_conv3")
gradcam = saliency * heatmap[..., np.newaxis]
save_img("guided_gradcam.jpg", deprocess_image(gradcam.numpy()))

saliency = get_class_saliency(guided_model,preprocessed_input,predicted_class)
gradcam = saliency * heatmap[..., np.newaxis]
save_img("guided_gradcam_class.jpg", deprocess_image(gradcam.numpy()))

