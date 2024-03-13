import tensorflow as tf


def loss_function(y_pred, margin=1.0):
    """
    This nested function calculates the loss for a given batch.
    """
    anchor, positive, negative = tf.split(y_pred, 3, axis=0)
    ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
    an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)

    losses = tf.maximum(0.0, ap_distance - an_distance + margin)
    return tf.reduce_mean(losses)
