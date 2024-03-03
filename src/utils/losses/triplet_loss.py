import tensorflow as tf

def kl_divergence(p, q):
  """
  This function calculates the KL divergence between two probability distributions.

  Args:
      p: A tensor representing the first probability distribution.
      q: A tensor representing the second probability distribution.

  Returns:
      A tensor representing the KL divergence between p and q.
  """
  eps = tf.keras.backend.epsilon()  # Add a small value to avoid log(0)
  return tf.reduce_mean(p * tf.math.log(p / (q + eps)))


def loss_function(y_pred, margin=1.0):
    """
    This nested function calculates the loss for a given batch.
    """
    anchor, positive, negative = tf.split(y_pred, 3, axis=0)
    ap_distance = kl_divergence(anchor, positive)
    an_distance = kl_divergence(anchor, negative)

    losses = tf.maximum(0.0, ap_distance - an_distance + margin)
    return tf.reduce_mean(losses)
