import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import (
    array_ops,
    control_flow_ops,
    math_ops,
    state_ops,
)


class PoincareSGD(tf.keras.optimizers.SGD):
    def __init__(self, learning_rate, **kwargs):
        super().__init__(learning_rate, name="PoincareSGD", **kwargs)

    @staticmethod
    def _euclid_to_riemann_grad(g, v):
        # eq (4) from Nickel without projection.
        theta_norm_sq = math_ops.reduce_sum(v * v, axis=1)[:, tf.newaxis]
        conversion_coef = ((1.0 - theta_norm_sq) ** 2) / 4.0
        return conversion_coef * g

    @staticmethod
    def proj(theta, eps=1e-2):
        # eq (3.5)
        norms = tf.reshape(linalg_ops.norm(theta, axis=1), (-1, 1))
        normed = theta / (norms + eps)
        return array_ops.where(norms < 1.0 - eps, theta, normed)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # intercept and scale gradient
        var_dtype = var.dtype.base_dtype
        lr = self._get_hyper("lr", var_dtype)
        grad = self._euclid_to_riemann_grad(grad, var)
        # grad = tf.clip_by_value(grad, -5., 5.)
        var_t = self.proj(var - lr * grad)
        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking
        )
        updates = [var_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        # intercept and scale gradient
        var_dtype = var.dtype.base_dtype
        lr = self._get_hyper("lr", var_dtype)
        grad = self._euclid_to_riemann_grad(grad, var)
        # grad = tf.clip_by_value(grad, -5., 5.)
        var_t = self.proj(var - self.learning_rate * grad)
        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking
        )
        updates = [var_update]
        return control_flow_ops.group(*updates)


class LorentzSGD(tf.keras.optimizers.SGD):
    def __init__(self, learning_rate, latent_dim):
        super().__init__(learning_rate, name="lorentzSGD")

    def lorentz_scalar_product(self, x, y):
        # eq (2)
        p = x * y
        dotL = -1.0 * p[:, 0]
        dotL += math_ops.sum(p[:, 1:], axis=1)
        return dotL

    def exp_map(v, x):
        # eq (9), v is the gradient
        vnorm = math_ops.sqrt(lorentz_scalar_product(v, v))
        return math_ops.cosh(vnorm) * x + math_ops.sinh(vnorm) * v / vnorm

    def proj(u, x):
        # eq (10.5), u is scaled gradient here
        return u + self.lorentz_scalar_product(x, u) * x

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr = self._get_hyper("learning_rate", var_dtype)
        gl = array_ops.identity(grad)
        gl[0, 0] = -1
        h = gl * grad
        gradfvar = self.proj(h, var)
        var_t = self.exp_map(-self.learning_rate * gradfvar, var)
        var_update = state_ops.assign(
            var, var_t, use_locking=self._use_locking
        )
        updates = [var_update]
        return control_flow_ops.group(*updates)
