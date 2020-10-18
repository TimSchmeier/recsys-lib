import tensorflow as tf
import numpy as np


class LorentzInitializer(tf.keras.initializers.Initializer):
    def __init__(self, low=-0.001, high=0.001):
        self.low = low
        self.high = high

    def __call__(self, shape, dtype=tf.float32):
        def set_x0(x):
            # eq (6)
            x0 = tf.expand_dims(
                tf.math.sqrt(1.0 + tf.linalg.norm(x[:, 1:], axis=-1)), axis=1
            )
            return tf.concat([x0, x[:, 1:]], axis=-1)

        theta = tf.random.uniform(
            minval=self.low, maxval=self.high, shape=shape
        )
        theta = tf.Variable(set_x0(theta), trainable=True, name="theta")
        return theta
