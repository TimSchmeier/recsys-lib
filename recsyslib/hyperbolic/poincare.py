import tensorflow as tf
from recsyslib.modelmixin import ModelMixin
import tensorflow.keras.backend as K


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
        self.r = K.variable(0.1)
        self.t = K.variable(0.1)
        self.theta = tf.keras.layers.Embedding(
            self.num_items,
            self.latent_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.001, maxval=0.001
            ),
            name="theta",
        )

    @staticmethod
    def distance(u, v, eps=1e-5):
        # eq (1)
        one_minus_u_norm_sq = 1.0 - tf.clip_by_value(
            tf.reduce_sum(u * u, axis=-1), 0, 1 - eps
        )
        one_minus_v_norm_sq = 1.0 - tf.clip_by_value(
            tf.reduce_sum(v * v, axis=-1), 0, 1 - eps
        )
        u_minus_v_norm_sq = tf.sqrt(
            tf.reduce_sum(tf.pow(u - v, 2), axis=-1) + eps
        )
        return tf.math.acosh(
            1.0
            + 2.0
            * u_minus_v_norm_sq
            / (one_minus_u_norm_sq * one_minus_v_norm_sq)
        )

    @staticmethod
    def loss_fxn(dist_uv, dist_uvprimes):
        # eq (5)
        return tf.reduce_mean(
            tf.math.log(
                tf.squeeze(tf.exp(-dist_uv))
                / tf.reduce_mean(tf.exp(-dist_uvprimes), axis=1)
            )
        )

    def fermi_dirac(self, duv):
        # eq (6)
        return 1.0 / (tf.math.exp((duv - self.r) / self.t) + 1.0)

    def score_is_a(self, u, v, alpha=1e3):
        # eq (7)
        return -(
            1.0
            + alpha * (tf.linalg.norm(v, axis=-1) - tf.linalg.norm(u, axis=-1))
        ) * self.distance(u, v)

    @tf.function
    def call(self, inputs):
        """Embed items into poincare ball.

        Args:
            inputs (tuple): (u[Int], v[Int], vprimes[List[Int]])
        """
        u, v = inputs
        u, v = self.theta(u), self.theta(v)
        duv = self.distance(u, v)
        return self.fermi_dirac(duv)
