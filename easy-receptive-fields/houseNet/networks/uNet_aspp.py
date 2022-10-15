from keras import Input, Model
from keras.layers import ZeroPadding2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2D

from networks.blocks.aspp_block import aspp_block
from networks.blocks.conv_block import convolution_block


def UNet_ASPP():
    """ UNet-like architecture with three pooling steps.
    The lowest two convolutions have been replaced with an aspp block.
    The amount of filters in some layers have been lowered in comparison to the UNet3 architecture
    to achieve a similar parameter size as the other networks.
    This network processes data with an input shape of (150, 150, 3) and an output shape of (150, 150, 1).

    # Returns:
        A Keras Model
    """
    input_layer = Input(shape=(150, 150, 3), name='input_layer')
    input_padded = ZeroPadding2D(name='input_padded')(input_layer)
    block_top = convolution_block(input_layer=input_padded, filters=32, name='block_top')
    pool_top = MaxPooling2D(pool_size=2, name='pool_top')(block_top)

    block_mid = convolution_block(input_layer=pool_top, filters=64, name='block_mid')
    pool_mid = MaxPooling2D(pool_size=2, name='pool_mid')(block_mid)
    pool_mid_padded = ZeroPadding2D(name='pool_mid_padded')(pool_mid)

    block_bot = convolution_block(input_layer=pool_mid_padded, filters=128, name='block_bot')
    pool_bot = MaxPooling2D(pool_size=2, name='pool_bot')(block_bot)

    block_aspp = aspp_block(input_layer=pool_bot, filters=128)

    # Up
    vert_up = UpSampling2D(size=2, name='vert_up')(block_aspp)
    conc_vert_bot = Concatenate(name='conc_vert_bot')([vert_up, block_bot])
    block_bot_up = convolution_block(input_layer=conc_vert_bot, filters=128, last_padding_valid=True,
                                     name='block_bot_up')

    bot_up = UpSampling2D(size=2, name='bot_up')(block_bot_up)
    conc_bot_mid = Concatenate(name='conc_bot_mid')([bot_up, block_mid])
    block_mid_up = convolution_block(input_layer=conc_bot_mid, filters=128, name='block_mid_up')

    mid_up = UpSampling2D(size=2, name='mid_up')(block_mid_up)
    conc_mid_top = Concatenate(name='conc_mid_top')([mid_up, block_top])
    block_top_up = convolution_block(input_layer=conc_mid_top, filters=128, last_padding_valid=True,
                                     name='block_top_up')

    output_layer = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='output_layer')(block_top_up)
    return Model(input_layer, output_layer)
