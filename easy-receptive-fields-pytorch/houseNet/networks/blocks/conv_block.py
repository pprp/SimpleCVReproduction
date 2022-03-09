from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization


def convolution_block(input_layer: Layer,
                      filters: int,
                      name: str,
                      kernel_size: int = 3,
                      activation: str = 'relu',
                      amount: int = 2,
                      padding: str = 'same',
                      dilation_rate: int = 1,
                      last_padding_valid: bool = False):
    """Generates a block of consecutive convolutional layers with batch normalization after each convolution.

    # Arguments:
        input_layer: Layer
            input for the first convolutional layer
        filters: integer
            amount of convolutional filters
        name: string
            prefix for the layer names
        kernel_size: integer
            size of the convolutional filter kernels, only quadratic kernels are allowed
        activation: string
            activation after each convolutional layer (see Keras documentation)
        amount: integer
            amount of consecutive layers
        padding: string
            padding for the input (either same or valid, see Keras documentation)
        dilation_rate: integer
            amount of dilation between filter weights
        last_padding_valid: boolean
            Last layer has valid padding and output will be smaller according to the kernel size

    # Returns:
        A Keras Layer
    """
    block = input_layer
    for i in range(amount):
        if i == amount - 1 and last_padding_valid:
            padding = 'valid'

        block = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       activation=activation,
                       padding=padding,
                       dilation_rate=dilation_rate,
                       name=name + '_' + str(i))(block)
        block = BatchNormalization(name=name + '_' + str(i) + '_norm')(block)
    return block
