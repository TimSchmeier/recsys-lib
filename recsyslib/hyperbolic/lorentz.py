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
            embeddings_initializer=LorentzInitializer(),
        )
        self.r = tf.Variable(0.1)
        self.t = tf.Variable(0.1)
        self.gl = tf.constant(
            tf.linalg.diag(
                tf.concat(
                    [-tf.ones((1,)), tf.ones(self.latent_dim - 1)], axis=-1
                )
            )
        )

    def lorentz_scalar_product(self, x, y):
        # eq (2)
        p = x * y
        s = -p[:, 0] + tf.reduce_sum(p[:, 1:], axis=-1)
        return tf.expand_dims(s, 1)

    def lorentz_scalar_product_batch(self, x, y):
        # for in-batch softmax
        inner = x[:, 1:] @ tf.transpose(y[:, 1:])
        outer = x[:, 0] * y[:, 0]
        return inner - outer

    def distance(self, x, y):
        # eq (5)
        return tf.math.acosh(
            tf.clip_by_value(
                -self.lorentz_scalar_product(x, y), 1.0 + 1e-6, 100
            )
        )

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
        return x[:, 1:] / np.expand_dims(x[:, 0] + 1.0, -1)

    def exp_map(self, v, x):
        """Maps a vector in the tangent space TxM onto the manifold M.
        Args:
            v (tensor): Riemannian gradient of f(x), result of proj(h, x).
            x (tensor): params to be updated.
        Returns (tensor): updated x."""
        lsp = tf.clip_by_value(self.lorentz_scalar_product(v, v), 1e-6, 100)
        vnorm = tf.math.sqrt(lsp)
        vnormed = v / vnorm
        return tf.math.cosh(vnorm) * x + tf.math.sinh(vnorm) * (v / vnorm)

    def proj(self, u, x):
        """Maps a vector in Euclidean space onto the tangent space Txm. eq (10.5)
        Args:
            u (tensor): euclidean gradient scaled by the Riemannian metric g.
            x (tensor): params to be updated.
        Returns (tensor): v, a vector tangent to x in Lorentzian space."""
        return u + self.lorentz_scalar_product(x, u) * x

    @tf.function
    def call(self, inputs):
        """Embed into lorentz space.
        Args:
            inputs (Array[Int], Array[Int]) : embedding ids
        """
        u, v = inputs
        u, v = self.theta(u), self.theta(v)
        duv = self.distance(u, v)
        return self.fermi_dirac(duv)

    # @tf.function
    def train_step(self, inputs):
        x, y, sample_weights = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compiled_loss(y, y_pred)
            grads = tape.gradient(loss, self.trainable_variables)
        remaining_grads_and_vars = []
        for grad, var in zip(grads, self.trainable_variables):
            if isinstance(
                grad, tf.python.framework.indexed_slices.IndexedSlices
            ):
                # these are embedding slices
                h = grad.values @ self.gl
                thetas = tf.gather(var, grad.indices)
                v = self.proj(h, thetas)
                new_thetas = self.exp_map(
                    -self.optimizer._hyper["learning_rate"] * v, thetas
                )
                indices = tf.expand_dims(grad.indices, -1)
                # manually update embedding tensors
                var_t = tf.tensor_scatter_nd_update(var, indices, new_thetas)
                self.theta.variables[0].assign(var_t)
            else:
                # pass other params to optimizer
                remaining_grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(remaining_grads_and_vars)
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {"loss": loss}
        metrics.update({m.name: m.result() for m in self.metrics})
        return metrics
