import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from get_data import DataGetter
from modelmixin import ModelMixin


class MatrixFactorization(ModelMixin, keras.Model):
    def __init__(self, num_users, num_items, biases=True, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.use_biases = biases
        self.user_embedding = layers.Embedding(
            self.num_users,
            self.latent_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_embedding = layers.Embedding(
            self.num_items,
            self.latent_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        if self.use_biases:
            self.user_bias = layers.Embedding(num_users, 1)
            self.item_bias = layers.Embedding(num_items, 1)

    @tf.function
    def call(self, inputs):
        user, item = inputs
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        dot_product = tf.reduce_sum(
            tf.multiply(user_vector, item_vector), axis=1
        )
        if self.use_biases:
            user_bias = self.user_bias(user)
            item_bias = self.item_bias(item)
            dot_product = dot_product + user_bias + item_bias
        return tf.nn.sigmoid(dot_product)


if __name__ == "__main__":
    d = DataGetter()
    df = d.get_ml_data()
    df = d.assign_indices(df)
    df = d.scale_rating(df)
    movieId_to_idx = d.get_item_idx_map(df)
    userId_to_idx = d.get_user_idx_map(df)
    (X_train, X_test, y_train, y_test) = d.split_data(df)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    mf = MatrixFactorization(num_users, num_movies, name="mf_sgd")

    mf.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
    )

    history = mf.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=5,
        verbose=1,
        validation_data=(X_test, y_test),
    )
