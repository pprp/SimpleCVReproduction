from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, UpSampling2D

from networks.blocks.conv_block import convolution_block


def DNet():
    """Fully Convolutional Neural Network which uses dilated convolutions instead of pooling
    to increase the perceptive field without decreasing the resolution for the subsequent layers.
    This network processes data with an input shape of (150, 150, 3) and an output shape of (150, 150, 1).

    # Returns:
        A Keras Model
    """

    input_layer = Input(shape=(150, 150, 3), name='input_layer')
    input_padded = ZeroPadding2D(name='input_padded')(input_layer)

    block_1 = convolution_block(input_layer=input_padded, filters=32, name='block_1')
    block_1_down = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                          strides=2, name='block_1_down')(block_1)

    block_2 = convolution_block(input_layer=block_1_down, filters=64, name='block_2')
    block_2_down = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                          dilation_rate=2, name='block_2_down')(block_2)

    block_3 = convolution_block(input_layer=block_2_down, filters=128, name='block_3')
    block_3_down = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                          dilation_rate=4, name='block_3_down')(block_3)

    block_4 = convolution_block(input_layer=block_3_down, filters=256, name='block_4')
    block_4_up = UpSampling2D(size=2, interpolation='bilinear', name='block_4_up')(block_4)

    block_output = Conv2D(filters=256, kernel_size=3, activation='relu', name='block_output')(block_4_up)
    block_output_norm = BatchNormalization(name='block_output_norm')(block_output)

    output_layer = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='output_layer')(block_output_norm)
    return Model(input_layer, output_layer)
