import configparser
import numpy as np

from keras.metrics import binary_accuracy
from keras.optimizers import Adam

from networks.trainNetwork import train_network
from networks.uNet3 import UNet3
from util.losses import dice
from util.metrics import intersection_over_union

config = configparser.ConfigParser()
config.read('config.ini')
weights_file_name = 'myUNet'

net = UNet3()
net.compile(optimizer=Adam(), loss=dice, metrics=[intersection_over_union, binary_accuracy])
train_network(model=net,
              path_config=config,
              weights_file_name=weights_file_name,
              batch_size=8,
              train_random_state=np.random.RandomState(2009),
              val_random_state=np.random.RandomState(2013),
              epochs=50,
              checkpoint_period=10,
              verbose=2)
