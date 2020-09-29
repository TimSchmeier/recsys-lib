import numpy as np
from recsyslib.als.alsmixin import ALSMixin
import tensorflow as tf
from recsyslib.target_transforms import interact_to_confidence


class ImplicitMF(ALSMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="impALS",
        alpha=40,
        lambduh=100,
        log=False,
        **kwargs,
    ):
        """Implimentation of:
        Y. Hu, Y. Koren and C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets",
        2008 Eighth IEEE International Conference on Data Mining, Pisa, 2008, pp. 263-272, doi: 10.1109/ICDM.2008.22.

        This implimentation uses tensorflow for faster compiled math operations, not optimization.
        These are multiplicative updates as in the original paper.

        Args:
            num_users (int): number of users
            num_items (int): number of items
            alpha (float): used to compute confidence value
            lambduh (float): L2 regularization strength
            log (bool): if True, confidence computed as 1.0 + alpha * log(1 + interact); else alpha * interact
        """
        self.name = name
        super().__init__(num_users, num_items, log, **kwargs)
        self.alpha = np.float32(alpha)
        self.lambduh = np.float32(lambduh)
        self.Y = tf.Variable(
            tf.random.normal(shape=[self.num_items, self.latent_dim]),
            trainable=False,
        )
        self.X = tf.Variable(
            tf.random.normal(shape=[self.num_users, self.latent_dim]),
            trainable=False,
        )

    @property
    def item_embeddings(self):
        return self.Y.numpy()

    @property
    def user_embeddings(self):
        return self.X.numpy()

    def call(self, inputs):
        """Run ALS.

        Args:
            inputs (tf.sparse.SparseTensor): confidence matrix
        """
        C = inputs
        self.update_items(C)
        self.update_users(C)

    @tf.function
    def vector_update(self, XtX, X, v, lambduhI):
        # eq (4) and (5)
        Cv = tf.linalg.diag(v)
        pv = tf.expand_dims(tf.where(v > 0.0, 1.0, 0.0), -1)
        CvminusI = Cv - tf.linalg.eye(tf.shape(v)[0])
        XtCiX = XtX + tf.transpose(X) @ CvminusI @ X
        inv = tf.linalg.inv(XtCiX + lambduhI)
        return tf.squeeze(inv @ tf.transpose(X) @ Cv @ pv)

    @tf.function
    def update_items(self, C):
        XtX = tf.transpose(self.X) @ self.X
        lambduhI = tf.linalg.eye(self.latent_dim) * self.lambduh
        for i in range(self.num_items):
            iv = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(C, [0, i], [self.num_users, 1])
                )
            )
            self.Y[i, :].assign(self.vector_update(XtX, self.X, iv, lambduhI))

    @tf.function
    def update_users(self, C):
        YtY = tf.transpose(self.Y) @ self.Y
        lambduhI = tf.linalg.eye(self.latent_dim) * self.lambduh
        for u in range(self.num_users):
            uv = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(C, [u, 0], [1, self.num_items])
                )
            )
            self.X[u, :].assign(self.vector_update(YtY, self.Y, uv, lambduhI))

    def transform_y(self, y):
        return interact_to_confidence(y, alpha=self.alpha)

    @tf.function
    def mse(self, M):
        return tf.reduce_mean(tf.pow(M - self.X @ tf.transpose(self.Y), 2))
