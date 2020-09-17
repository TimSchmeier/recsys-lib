import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from get_data import DataGetter
from mf_sgd import MatrixFactorization


class NCS(MatrixFactorization):
    def __init__(self, num_users, num_items, num_dense=3, **kwargs):
        super().__init__(num_users, num_items, biases=False, **kwargs)
        self.num_dense = num_dense
        self.dense_layers = [
            layers.Dense((self.latent_dim * 2) / (l ** 2), activation="relu")
            for l in range(1, num_dense)
        ]
        self.output_layer = layers.Dense(1)

    @tf.function
    def call(self, inputs):
        user, item = inputs
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        x = tf.concat([user_vector, item_vector], axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        logit = self.output_layer(x)
        return tf.nn.sigmoid(logit)


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

    ncs = NCS(num_users, num_movies, name="ncs")

    ncs.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
    )

    history = ncs.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=5,
        verbose=1,
        validation_data=(X_test, y_test),
    )
