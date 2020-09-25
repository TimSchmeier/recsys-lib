import tensorflow as tf
from recsyslib.modelmixin import ModelMixin
from recsyslib.keras_extended.initializers import LorentzInitializer


class LorentzEmbedding(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        """Implimentation of:
            Maximilian Nickel, Douwe Kiela. 2018.
            Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. ICML.

        Args:
            num_items (int): number of items to embed.
        """
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.EPS = 1e-5
        self.theta = tf.keras.layers.Embedding(
            self.num_items,
            self.latent_dim,
            kernel_initializer=LorentzInitializer(),
        )

    def lorentz_scalar_product(self, x, y):
        # eq (2)
        p = x * y
        dotL = -1.0 * p[:, 0]
        dotL += tf.reduce_sum(p[:, 1:], axis=1)
        return dotL

    def dist(self, x, y):
        # eq (5)
        return tf.math.arcosh(-1.0 * self.lorentz_scalar_product(x, y))

    def fermi_dirac(self, duv):
        # eq (6)
        return 1.0 / (tf.math.exp((duv - self.r) / self.t) + 1.0)

    @staticmethod
    def ranking_loss(dist_uv, dist_uvprimes):
        # eq (5)
        return tf.log(
            tf.exp(-dist_uv) / tf.reduce_mean(tf.exp(-dist_uvprimes))
        )

    @staticmethod
    def to_poincare(x):
        # eq (11)
        return x[:, 1:] / (x[:, 0] + 1.0)

    @tf.function
    def call(self, inputs):
        """Embed into lorentz space.
        Args:
            inputs (tuple):
        """
        u, v = inputs
        u, v = self.theta(u), self.theta(v)
        duv = self.distance(u, v)
        return self.fermi_dirac(duv)
