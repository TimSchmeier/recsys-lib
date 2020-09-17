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


WINDOW_SIZE = 2
NEG_SAMPLES = 2
BATCH_SIZE = 32


class W2V(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.hidden = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )
        self.context = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )

    def call(self, inputs):
        word, context = inputs
        w = self.hidden(word)
        c = self.context(context)
        dot_product = tf.keras.layers.dot([w, c], axes=(1))
        return tf.nn.sigmoid(dot_product)

    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            ypred = self.call(x)
            loss = tf.reduce_mean(binary_crossentropy(y, ypred))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, ypred)
        return {m.name: m.result() for m in self.metrics}


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

    m = W2V(NMOVIES, name="w2v")

    m.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=["AUC"])

    m.fit(sgdg, epochs=1)

    """
    opt = tf.keras.optimizers.Adam(lr=0.001)
    for minibatch in range(len(sgdg)):
	x, y = sgdg[minibatch]
	with tf.GradientTape() as tape:
            ypred = m(x)
            loss = tf.reduce_mean(binary_crossentropy(y, ypred))
	grads = tape.gradient(loss, m.trainable_variables)
	opt.apply_gradients(zip(grads, m.trainable_variables))
    """
