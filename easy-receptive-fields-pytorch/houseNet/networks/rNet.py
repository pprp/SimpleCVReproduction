from keras import Input, Model
from keras.layers import Concatenate, Conv2D, ZeroPadding2D, BatchNormalization, UpSampling2D

from networks.blocks.conv_block import convolution_block
from networks.blocks.residual_block import res_block


def RNet():
    """Network which combines a network with residual blocks and a UNet.
    The ResNet structure is setup according to:

        Fisher Yu, Vladlen Koltun, Thomas Funkhouser
        Dilated Residual Networks
        arXiv:1705.09914, 2017

    The number of filters are adapted to achieve a similar parameter size as the other networks.
    This network processes data with an input shape of (150, 150, 3) and an output shape of (150, 150, 1).

    # Returns:
        A Keras Model
    """

    input_layer = Input(shape=(150, 150, 3), name='input_layer')
    input_padded = ZeroPadding2D(name='input_padded')(input_layer)
    conv_1 = Conv2D(filters=16, kernel_size=7, activation='relu', padding='same', name='conv_1')(input_padded)
    conv_1_norm = BatchNormalization(name='conv_1_norm')(conv_1)

    res_block_1 = res_block(conv_1_norm, filters=16, name='res_block_1')
    down_1 = Conv2D(filters=16, kernel_size=3, strides=2, activation='relu', padding='same', name='down_1')(res_block_1)
    down_1_norm = BatchNormalization(name='down_1_norm')(down_1)

    res_block_2 = res_block(down_1_norm, filters=32, name='res_block_2')
    down_2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same', name='down_2')(res_block_2)
    down_2_norm = BatchNormalization(name='down_2_norm')(down_2)

    res_block_3 = res_block(down_2_norm, filters=64, name='res_block_3')
    res_block_4 = res_block(res_block_3, filters=64, name='res_block_4')

    down_3 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same', name='down_3')(res_block_4)
    down_3_norm = BatchNormalization(name='down_3_norm')(down_3)

    res_block_5 = res_block(down_3_norm, filters=64, name='res_block_5')
    res_block_6 = res_block(res_block_5, filters=64, name='res_block_6')

    res_block_7 = res_block(res_block_6, filters=64, dilation_rate=2, name='res_block_7')
    res_block_8 = res_block(res_block_7, filters=64, dilation_rate=2, name='res_block_8')

    res_block_9 = res_block(res_block_8, filters=64, dilation_rate=4, name='res_block_9')
    res_block_10 = res_block(res_block_9, filters=64, dilation_rate=4, name='res_block_10')

    conv_block_1 = convolution_block(res_block_10, filters=64, dilation_rate=2, name='conv_block_1')
    conv_block_2 = convolution_block(conv_block_1, filters=64, name='conv_block_2')

    pre_up_1 = convolution_block(conv_block_2, filters=128, name='pre_up_1')
    up_1 = UpSampling2D(size=2, name='up_1')(pre_up_1)
    skip_1 = Concatenate(name='skip_1')([res_block_4, up_1])

    pre_up_2 = convolution_block(skip_1, filters=128, name='pre_up_2')
    up_2 = UpSampling2D(size=2, name='up_2')(pre_up_2)
    skip_2 = Concatenate(name='skip_2')([res_block_2, up_2])

    pre_up_3 = convolution_block(skip_2, filters=128, name='pre_up_3')
    up_3 = UpSampling2D(size=2, name='up_3')(pre_up_3)
    skip_3 = Concatenate(name='skip_3')([res_block_1, up_3])

    block_last = convolution_block(skip_3, filters=64, name='block_last', last_padding_valid=True)
    output = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='output')(block_last)

    return Model(input_layer, output)
