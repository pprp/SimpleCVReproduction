from keras.engine import Layer
from keras.layers import Conv2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization


def aspp_block(input_layer: Layer,
               filters: int,
               kernel_size: int = 3,
               activation: str = 'relu',
               pyramid_size: int = 4,
               exponential_dilation: bool = False):
    """Generates an aspp (atrous spatial pyramid pooling) block, as described in

        Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
        Rethinking Atrous Convolution for Semantic Image Segmentation,
        arXiv:1706.05587v3, 2017

    # Arguments:
        input_layer: Layer
            input for the first convolutional layer
        filters: integer
            amount of convolutional filters
        kernel_size: integer
            size of the convolutional filter kernels, only quadratic kernels are allowed
        activation: string
            activation after each convolutional layer (see Keras documentation)
        pyramid_size: integer
            number of parallel convolutional layers
        exponential_dilation: boolean
            dilation rate increases exponentially (basis 2), default is incremental

    # Returns:
        A Keras Layer
    """
    if exponential_dilation:
        dilation_rates = [2 ** x for x in range(pyramid_size)]
    else:
        dilation_rates = list(range(1, pyramid_size + 1))

    kernel = [1] + [kernel_size] * (pyramid_size - 1)

    pyramid = []
    for i in range(pyramid_size):
        p = Conv2D(filters=filters,
                   kernel_size=kernel[i],
                   activation=activation,
                   padding='same',
                   dilation_rate=dilation_rates[i],
                   name='aspp_' + str(i + 1))(input_layer)
        p = BatchNormalization(name='aspp_' + str(i + 1) + '_bn')(p)
        pyramid.append(p)

    avg = AveragePooling2D(pool_size=2, name='aspp_avg_pool')(input_layer)
    avg = Conv2D(filters=filters, kernel_size=1, activation=activation, name='aspp_avg_1x1')(avg)
    avg = BatchNormalization(name='aspp_avg_bn')(avg)
    avg_features = UpSampling2D(size=2, interpolation='bilinear', name='aspp_avg_features')(avg)

    aspp = Concatenate(name='aspp_conc')(pyramid + [avg_features])
    aspp = Conv2D(filters=filters, kernel_size=1, activation=activation, name='aspp_1x1')(aspp)
    aspp = BatchNormalization(name='aspp_out')(aspp)
    return aspp
