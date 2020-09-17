import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import (
    make_sampling_table,
    skipgrams,
)
from tensorflow.keras.preprocessing.text import (
    Tokenizer,
    text_to_word_sequence,
)
from tensorflow.keras.losses import binary_crossentropy
from random import shuffle
from tensorflow.keras.utils import Sequence
import math
import numpy as np
from get_data import DataGetter
from modelmixin import ModelMixin
from kerasutils.optimizers import rSGD
from kerasutils.callbacks import BurnIn


WINDOW_SIZE = 2
NEG_SAMPLES = 10
BATCH_SIZE = 32


class PoincareEmbedding(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.EPS = 1e-5
        self.theta = tf.keras.layers.Embedding(
            self.num_items,
            self.latent_dim,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.001
            ),
            # constraints=tf.keras.constraints.max_norm(
            #    max_value=1.0 - self.EPS, axis=1
            # )
        )

    @staticmethod
    def distance(u, v):
        # eq (1)
        one_minus_u_norm_sq = 1 - tf.linalg.norm(u, axis=1) ** 2
        one_minus_v_norm_sq = 1 - tf.linalg.norm(v, axis=1) ** 2
        u_minus_v_norm_sq = tf.linalg.norm(u - v, axis=1) ** 2
        return tf.math.acos(
            1
            + 2
            * u_minus_v_norm_sq
            / (one_minus_u_norm_sq * one_minus_v_norm_sq)
        )

    @staticmethod
    def loss(dist_uv, dist_uvprimes):
        # eq (5)
        return tf.log(
            tf.exp(-dist_uv) / tf.reduce_mean(tf.exp(-dist_uvprimes))
        )

    @staticmethod
    def project(theta, eps=1e-5):
        # eq (3.5)
        norms = tf.linalg.norm(theta, axis=1)
        theta = tf.where(norms < 1, theta, theta / norms + eps)
        return theta

    def call(self, inputs):
        parent, child, unrelated = inputs
        u = self.theta[parent]
        dist_uv = self.distance(u, self.theta[child])
        dist_uvprimes = self.distance(u, self.theta[unrelated])
        return dist_uv, dist_uvprimes

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            dist_uv, dist_uvprimes = self.call(inputs)
            l = self.loss(dist_uv, dist_uvprimes)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.theta = self.project(self.theta)


class SkipGramDataGenerator(Sequence):
    def __init__(
        self,
        corpus,
        vocabsize,
        window_size=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        negative_samples=NEG_SAMPLES,
        sampling_table=None,
    ):
        self.corpus = corpus
        self.batch_size = batch_size
        self.vocabsize = vocabsize
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.sampling_table = sampling_table
        self.X, self.y = self.skipgramgen(self.corpus)

    def __getitem__(self, index):
        index = np.random.choice(len(self.y), self.batch_size)
        return (
            (self.X[index, 0], self.X[index, 1]),
            np.expand_dims(self.y[index], -1),
        )

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def skipgramgen(self, corpus):
        pairs, labels = [], []
        for doc in corpus:
            p, l = skipgrams(
                doc,
                self.vocabsize,
                window_size=self.window_size,
                negative_samples=self.negative_samples,
                sampling_table=self.sampling_table,
            )
            pairs.extend(p)
            labels.extend(l)
        return np.asarray(pairs), np.asarray(labels)


if __name__ == "__main__":
    d = DataGetter()
    df = d.get_ml_data()
    NMOVIES = len(df["movieId"].unique())
    sequences = d.get_item_sequences(df)

    tokenizer = Tokenizer(NMOVIES, oov_token="OOV")
    tokenizer.fit_on_texts(sequences)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    train_sequences = tokenizer.texts_to_sequences(sequences)
    sampling_table = make_sampling_table(NMOVIES)
    sgdg = SkipGramDataGenerator(
        train_sequences, NMOVIES, sampling_table=sampling_table
    )

    m = PoincareEmbedding(NMOVIES, name="poincare")

    lr = 0.001
    m.compile(optimizer=rSGD(lr=lr), metrics=["AUC"])

    m.fit(sgdg, epochs=1, callbacks=[BurnIn(lr / 10.0)])
