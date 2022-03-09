import configparser

from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta


class LearningRateTracker(Callback):
    """ Tracks the current learning rate and prints it after each epoch.
    This tracker actually just performs the Keras calculation.
    """

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = optimizer.lr
        if optimizer.initial_decay > 0:
            lr = lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations,
                                                           K.dtype(optimizer.decay))))

        if type(optimizer) in [SGD, RMSprop, Adagrad, Adadelta]:
            print('Current learning rate: %f' % K.eval(lr))

        elif type(optimizer) == Adam:
            t = K.cast(optimizer.iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(optimizer.beta_2, t)) /
                         (1. - K.pow(optimizer.beta_1, t)))
            print('Current learning rate: %f' % K.eval(lr_t))

        else:
            print('Can not print lr: Optimizer not supported.')


def setup_callbacks(path_config: configparser.ConfigParser,
                    weights_file_name: str,
                    checkpoint_period: int = 0,
                    use_lr_reduction: bool = False):
    """ Generates a list of useful callbacks.
    # Arguments:
        path_config: ConfigParser
            configuration which should contain absolute paths to models and logs
        weights_file_name: string
            filename prefix for the saved weights
        checkpoint_period: integer
            period in which the model gets saved
        use_lr_reduction: boolean
            learning rate will be lowered when iou did not increase sufficiently for a while
    """

    log_dir = path_config['DIRECTORIES']['logs']
    model_dir = path_config['DIRECTORIES']['models']

    callbacks = []
    tensor_board = TensorBoard(log_dir=log_dir)
    learning_rate_tracker = LearningRateTracker()
    callbacks.append(tensor_board)
    callbacks.append(learning_rate_tracker)

    if checkpoint_period > 0:
        file_path = model_dir + weights_file_name + '.{epoch:02d}-{val_intersection_over_union:.4f}.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=file_path,
                                           monitor='val_intersection_over_union',
                                           verbose=1,
                                           mode='max',
                                           period=checkpoint_period)
        callbacks.append(model_checkpoint)

    if use_lr_reduction:
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_intersection_over_union',
                                               factor=1.0,
                                               patience=10,
                                               verbose=1,
                                               mode='max',
                                               min_delta=0.005)
        callbacks.append(reduce_lr_callback)

    return callbacks
