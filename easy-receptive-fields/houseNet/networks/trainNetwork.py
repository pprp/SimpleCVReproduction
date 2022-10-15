import configparser
import numpy as np

from keras import Model

from util.data_loader import DataLoader
from util.callbacks import setup_callbacks
from util.tf_utils import initialize_tf_variables


def train_network(model: Model,
                  path_config: configparser.ConfigParser,
                  weights_file_name: str,
                  batch_size: int = 32,
                  epochs: int = 50,
                  train_random_state: np.random.RandomState = None,
                  val_random_state: np.random.RandomState = None,
                  checkpoint_period: int = 10,
                  verbose: int = 1,
                  initial_epoch: int = 0):
    """Trains the model.

    # Arguments:
        model: Model
            Compiled Keras model
        path_config: ConfigParser
            configuration which contains absolute paths to train_data, val_data, models, logs
        weights_file_name: string
            filename for the saved weights
        pre_trained_model_file_name: string
            absolute path to an already trained model
        batch_size: integer
            amount of input images which are processed as a batch during learning
        epochs: integer
            Total amount of epochs which should be trained
        random_state: integer
            seed for the random state which will be used for data augmentation
        checkpoint_period: integer
            period in which the model gets saved
        verbose: integer
            verbosity level (0,1 or 2)
    """

    if not isinstance(model, Model):
        print('Model is wrong! Use a keras model for training.')
        return
    if not isinstance(path_config, configparser.ConfigParser):
        print('Path configuration is wrong! Use configparser.ConfigParser to read the configuration.')
        return

    print('Setting up data loader ...')
    train_data_loader = DataLoader(data_directory=path_config['DIRECTORIES']['train_data'],
                                   batch_size=batch_size,
                                   down_sample_factor=2,
                                   random_state=train_random_state)

    val_data_loader = DataLoader(data_directory=path_config['DIRECTORIES']['val_data'],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 augment=False,
                                 down_sample_factor=2,
                                 random_state=val_random_state)

    print('Setting up callbacks ...')
    callbacks = setup_callbacks(path_config, weights_file_name, checkpoint_period)

    if initial_epoch == 0:
        print('Initialize tf variables ...')
        initialize_tf_variables()

    print('Start training ...')
    model.fit_generator(generator=train_data_loader,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=val_data_loader,
                        workers=6,
                        use_multiprocessing=True,
                        initial_epoch=initial_epoch)
