import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from get_data import DataGetter
from modelmixin import ModelMixin
from scipy.sparse import csr_matrix


class Encoder(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, num_dense, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = []
        for layer in range(num_dense, 0, -1):
            self.dense_layers.append(
                layers.Dense(
                    self.latent_dim * (layer * 2),
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
        self.dense_out = layers.Dense(self.latent_dim)

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        encoded = self.dense_out(x)
        return encoded


class Decoder(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, num_dense, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = []
        for layer in range(1, num_dense + 1):
            self.dense_layers.append(
                layers.Dense(
                    self.latent_dim * (layer * 2),
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
        self.dense_out = layers.Dense(
            self.num_items, activation="sigmoid", name="out"
        )

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        recon_x = self.dense_out(x)
        return recon_x


class AE(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, num_dense=4, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.enc = Encoder(num_users, num_items, num_dense, name="enc")
        self.dec = Decoder(num_users, num_items, num_dense, name="dec")

    @tf.function
    def call(self, inputs):
        h = self.enc(inputs)
        return self.dec(h)
