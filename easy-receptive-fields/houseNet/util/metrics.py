import tensorflow as tf


def intersection_over_union(y_true, y_pred):
    """ Calculates intersection over union metric for two classes.
    """

    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
