import os
import pandas as pd
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
        for l in range(num_dense, 0, -1):
            self.dense_layers.append(
                layers.Dense(
                    self.latent_dim * (l * 2),
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
        self.dense_out = layers.Dense(self.latent_dim)

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
        for l in range(1, num_dense + 1):
            self.dense_layers.append(
                layers.Dense(
                    self.latent_dim * (l * 2),
                    activation="relu",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
        self.dense_out = layers.Dense(
            self.num_items, activation="sigmoid", name="out"
        )

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

    def call(self, inputs):
        h = self.enc(inputs)
        return self.dec(h)


if __name__ == "__main__":
    d = DataGetter()
    df = d.get_ml_data()
    # only keep good ratings
    df = df[df["rating"] >= 3]
    df = d.assign_indices(df)
    movieId_to_idx = d.get_item_idx_map(df)
    userId_to_idx = d.get_user_idx_map(df)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    user_by_item = csr_matrix(
        (np.ones(df.shape[0]), (df["user_idx"], df["movie_idx"])),
        shape=(num_users, num_movies),
    )

    ae = AE(num_users, num_movies, name="ae")

    ae.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    )

    num_epochs = 5
    history = ae.fit(
        x=user_by_item,
        y=user_by_item,
        batch_size=32,
        epochs=num_epochs,
        verbose=1,
    )
