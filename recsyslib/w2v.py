import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import (
    skipgrams,
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import Sequence
import math
import numpy as np
from recsyslib.modelmixin import ModelMixin


WINDOW_SIZE = 2
NEG_SAMPLES = 2
BATCH_SIZE = 32


class W2V(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        """Implimentation of:
            Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space".
            arXiv:1301.3781

        Args:
            num_items (int): total number of sequence items to embed.
            **kwargs
        """
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.hidden = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )
        self.context = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )

    @tf.function
    def call(self, inputs):
        """Embed word pairs.

        Args:
            inputs (tuple): (word, context)
        """
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
            pair, label = skipgrams(
                doc,
                self.vocabsize,
                window_size=self.window_size,
                negative_samples=self.negative_samples,
                sampling_table=self.sampling_table,
            )
            pairs.extend(pair)
            labels.extend(label)
        return np.asarray(pairs), np.asarray(labels)
