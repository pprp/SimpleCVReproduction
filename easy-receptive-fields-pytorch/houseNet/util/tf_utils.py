import tensorflow as tf
import keras.backend as K


def initialize_tf_variables():
    """Initializes tensorflow global/local variables.
    This is needed for custom metrics.
    """
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.local_variables_initializer())
