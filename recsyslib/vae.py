import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from get_data import DataGetter
from modelmixin import ModelMixin
from scipy.sparse import csr_matrix
import tensorflow.keras.backend as K
from ae import Decoder
from callbacks import AnnealKLloss


class Sampling(layers.Layer):
    def __init__(self):
        super().__init__(trainable=False, name="sampling")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEncoder(ModelMixin, tf.keras.Model):
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
        self.dense_mu = layers.Dense(self.latent_dim, name="z_mean")
        self.dense_var = layers.Dense(self.latent_dim, name="z_log_var")
        self.sample = Sampling()

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        z_mean, z_log_var = self.dense_mu(x), self.dense_var(x)
        z = self.sample((z_mean, z_log_var))
        return z_mean, z_log_var, z


class VAE(ModelMixin, tf.keras.Model):
    def __init__(
        self,
        num_users,
        num_items,
        num_dense=4,
        kl_weight=tf.Variable(0.0, dtype=tf.float32, trainable=False),
        **kwargs
    ):
        super().__init__(num_users, num_items, **kwargs)
        self.kl_weight = kl_weight
        self.enc = VAEncoder(num_users, num_items, num_dense, name="enc")
        self.dec = Decoder(num_users, num_items, num_dense, name="dec")

    def call(self, inputs):
        z_mean, z_log_var, z = self.enc(inputs)
        reconstructed_x = self.dec(z)
        return z_mean, z_log_var, z, reconstructed_x

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstructed_x = self.call(inputs)
            reconstruction_loss = (
                tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.sparse.to_dense(inputs), reconstructed_x
                    )
                )
                * self.num_items
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss_weighted = kl_loss * self.kl_weight  # annealing
            total_loss = reconstruction_loss + kl_loss_weighted
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "kl_weight": self.kl_weight,
            "kl_loss_weighted": kl_loss_weighted,
        }


if __name__ == "__main__":
    d = DataGetter()
    # only keep good ratings
    d.df = d.df[d.df["rating"] >= 3]
    movieId_to_idx = d.get_item_idx_map(df)
    userId_to_idx = d.get_user_idx_map(df)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    user_by_item = csr_matrix(
        (np.ones(d.df.shape[0]), (d.df["user_idx"], d.df["movie_idx"])),
        shape=(num_users, num_movies),
    )

    vae = VAE(num_users, num_movies, name="vae")

    vae.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))

    num_epochs = 5
    history = vae.fit(
        user_by_item[:128, :],
        batch_size=32,
        epochs=num_epochs,
        verbose=1,
        callbacks=[AnnealKLloss(num_epochs)],
    )
