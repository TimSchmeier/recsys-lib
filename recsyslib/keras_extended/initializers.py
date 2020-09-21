import tensorflow as tf
import numpy as np


class LorentzInitializer(tf.keras.initializers.Initializer):
    def __init__(self, low=-0.001, high=0.001):
        self.low = low
        self.high = high

    def __call__(self, shape, dtype=tf.float32):
        # eq (6)
        def set_x0(x):
            x0 = np.sqrt(1.0 + x[:, 1:])
            x[:, 0] = x0
            return x

        theta = np.random.uniform(self.low, self.high, size=shape)
        theta = tf.Variable(set_x0(theta), trainable=True, name="theta")
        return theta
