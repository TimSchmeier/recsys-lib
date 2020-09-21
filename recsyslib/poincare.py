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
from recsyslib.keras_extended.optimizers import PoincareSGD
from recsyslib.keras_extended.callbacks import BurnIn


WINDOW_SIZE = 2
NEG_SAMPLES = 10
BATCH_SIZE = 32


class PoincareEmbedding(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        """Implimentation of:
            Maximilian Nickel, Douwe Kiela. 2017.
            Poincaré Embeddings for Learning Hierarchical Representations. NIPS.

        Args:
            num_items (int): number of items to embed
            **kwargs
        """
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.EPS = 1e-5
        self.theta = tf.keras.layers.Embedding(
            self.num_items,
            self.latent_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.001, maxval=0.001
            ),
        )

    @staticmethod
    def distance(u, v, eps=1e-5):
        # eq (1)
        tf.debugging.check_numerics(u, "u")
        tf.debugging.check_numerics(v, "v")
        one_minus_u_norm_sq = 1.0 - tf.clip_by_value(
            tf.reduce_sum(u * u, axis=2), 0, 1 - eps
        )
        one_minus_v_norm_sq = 1.0 - tf.clip_by_value(
            tf.reduce_sum(v * v, axis=2), 0, 1 - eps
        )
        tf.debugging.check_numerics(one_minus_u_norm_sq, "one u")
        tf.debugging.check_numerics(one_minus_v_norm_sq, "one v")
        u_minus_v_norm_sq = tf.sqrt(
            tf.reduce_sum(tf.pow(u - v, 2), axis=2) + eps
        )
        tf.debugging.check_numerics(u_minus_v_norm_sq, "u minus v")
        toacosh = 1.0 + 2.0 * u_minus_v_norm_sq / (
            one_minus_u_norm_sq * one_minus_v_norm_sq
        )
        tf.debugging.check_numerics(toacosh, "toacosh")
        acosh = tf.math.acosh(toacosh)
        tf.debugging.check_numerics(acosh, "acosh")
        return acosh

    @staticmethod
    def loss_fxn(dist_uv, dist_uvprimes):
        # eq (5)
        return tf.reduce_mean(
            tf.math.log(
                tf.squeeze(tf.exp(-dist_uv))
                / tf.reduce_mean(tf.exp(-dist_uvprimes), axis=1)
            )
        )

    # @tf.function
    def call(self, inputs):
        """Embed items into poincare ball.

        Args:
            inputs (tuple): (u[Int], v[Int], vprimes[List[Int]])
        """
        # tf.print("start")
        _u, _v, _vprimes = (
            tf.reshape(inputs[:, 0], (-1, 1)),
            tf.reshape(inputs[:, 1], (-1, 1)),
            inputs[:, 2:],
        )  # indices
        # tf.print(_u.shape, _v.shape, _vprimes.shape)
        # tf.print(_u, _v, _vprimes)
        u, v, vprimes = self.theta(_u), self.theta(_v), self.theta(_vprimes)
        # tf.print(u.shape, v.shape, vprimes.shape)
        dist_uv = self.distance(u, v)
        tf.debugging.check_numerics(dist_uv, "dist_uv nans")
        dist_uvprimes = self.distance(u, vprimes)
        tf.debugging.check_numerics(dist_uvprimes, "dist_uvprime nans")
        return dist_uv, dist_uvprimes

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            dist_uv, dist_uvprimes = self.call(inputs)
            loss = self.loss_fxn(dist_uv, dist_uvprimes)
            tf.debugging.check_numerics(loss, "loss nans")
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        tf.debugging.check_numerics(
            self.theta.embeddings, "embeddings after grad update nans"
        )
        tf.Assert(
            tf.less_equal(tf.linalg.norm(self.theta.embeddings), 1.0),
            [tf.linalg.norm(self.theta.embeddings)],
            name="theta",
        )
        return {"loss": loss}


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
