import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import (
    make_sampling_table,
    skipgrams,
)
from tensorflow.keras.preprocessing.text import (
    Tokenizer,
)
from tensorflow.keras.utils import Sequence
import math
import numpy as np
from recsyslib.modelmixin import ModelMixin
from recsyslib.keras_extended.optimizers import LorentzSGD
from recsyslib.keras_extended.callbacks import BurnIn
from recsyslib.keras_extended.initializers import LorentzInitializer


WINDOW_SIZE = 2
NEG_SAMPLES = 10
BATCH_SIZE = 32


@tf.function
def lorentz_scalar_product(x, y):
    # eq (2)
    p = x * y
    dotL = -1.0 * p[:, 0]
    dotL += tf.reduce_sum(p[:, 1:], axis=1)
    return dotL


class LorentzEmbedding(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        """Implimentation of:
            Maximilian Nickel, Douwe Kiela. 2018.
            Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. ICML.

        Args:
            num_items (int): number of items to embed.
        """
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.EPS = 1e-5
        self.theta = tf.keras.layers.Embedding(
            self.num_items,
            self.latent_dim,
            kernel_initializer=LorentzInitializer(),
        )

    def dist(self, x, y):
        # eq (5)
        return tf.math.arcosh(-1.0 * lorentz_scalar_product(x, y))

    @staticmethod
    def loss(dist_uv, dist_uvprimes):
        # eq (5)
        return tf.log(
            tf.exp(-dist_uv) / tf.reduce_mean(tf.exp(-dist_uvprimes))
        )

    @staticmethod
    def to_poincare(x):
        # eq (11)
        return x[:, 1:] / (x[:, 0] + 1.0)

    @tf.function
    def call(self, inputs):
        """Embed into lorentz space.
        Args:
            inputs (tuple):
        """
        parent, child, unrelated = inputs
        u = self.theta(parent)
        dist_uv = self.distance(u, self.theta(child))
        dist_uvprimes = self.distance(u, self.theta(unrelated))
        return dist_uv, dist_uvprimes

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            dist_uv, dist_uvprimes = self.call(inputs)
            loss = self.loss(dist_uv, dist_uvprimes)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.theta = self.project(self.theta)
