import keras.backend as K


def dice(y_true, y_pred):
    """ Calculates the dice loss which can be used to maximize the intersection of union
    for image segmentation tasks.
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
