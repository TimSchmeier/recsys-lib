import tensorflow as tf


class rSGD(tf.keras.optimizers.SGD):
    def __init__(self, learning_rate):
        super().__init__(learning_rate, name="rSGD")

    @staticmethod
    def _euclid_to_riemann_grad(g, v):
        # eq (4) from Nickel without projection.
        theta_norm_sq = tf.linalg.norm(v, axis=1)[:, tf.newaxis] ** 2
        conversion_coef = ((1 - theta_norm_sq) ** 2) / 4
        return conversion_coef * g

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # intercept and scale gradient
        grad = self._euclid_to_riemann_grad(grad, var)
        return super()._resource_apply_dense(grad, var, apply_state)
