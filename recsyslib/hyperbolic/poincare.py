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

    @staticmethod
    def _euclid_to_riemann_grad(g, v):
        # eq (4) from Nickel without projection.
        theta_norm_sq = v * v
        conversion_coef = ((1.0 - theta_norm_sq) ** 2) / 4.0
        return conversion_coef * g

    @staticmethod
    def proj(theta, eps=1e-2):
        # eq (3.5)
        norms = tf.linalg.norm(theta)
        normed = theta / (norms + eps)
        return tf.where(norms < 1.0 - eps, theta, normed)

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
            inputs (tuple): (u[Int], v[Int])
        """
        u, v = inputs
        u, v = self.theta(u), self.theta(v)
        duv = self.distance(u, v)
        return self.fermi_dirac(duv)

    @tf.function
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
                thetas = tf.gather(var, grad.indices)
                grad = self._euclid_to_riemann_grad(grad, thetas)
                new_thetas = self.proj(
                    thetas - self.optimizer._hyper["learning_rate"] * grad
                )
                indices = tf.expand_dims(grad.indices, -1)
                # manual parameter updates
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
