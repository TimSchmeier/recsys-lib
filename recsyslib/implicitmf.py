import numpy as np
from recsyslib.modelmixin import ModelMixin
from tqdm import tqdm
import tensorflow as tf


class ImplicitALS(ModelMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="impALS",
        alpha=40,
        lambduh=100,
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
        """
        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.EPS = 1e-6
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

    def fit(self, x, y, epochs=20):
        """Run Implicit ALS on user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """
        users, items = x
        confidences = self.interact_to_confidence(y).astype(np.float32)
        indices = [[u, i] for u, i in zip(users, items)]
        C = tf.sparse.reorder(
            tf.sparse.SparseTensor(
                indices,
                confidences,
                dense_shape=[self.num_users, self.num_items],
            )
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(C)
            self.logger.info(f"epoch {it} complete")

    def interact_to_confidence(self, val):
        return self.alpha * val

    def loginteract_to_confidence(self, val):
        # eq (6)
        return 1.0 + self.alpha * np.log(1 + val / self.EPS)

    def call(self, inputs):
        """Run ALS.

        Args:
            inputs (tf.sparse.SparseTensor): confidence matrix
        """
        C = inputs
        self.update_items(C)
        self.logger.info("updated items")
        self.update_users(C)
        self.logger.info("updated users")

    @tf.function
    def vector_update(self, XtX, X, v, lambduhI):
        # eq (4) and (5)
        Cv = tf.linalg.diag(v)
        pv = tf.expand_dims(tf.where(v > 0.0, 1.0, 0.0), -1)
        CvminusI = Cv - tf.linalg.eye(tf.shape(v)[0])
        XtCiX = XtX + tf.transpose(X) @ CvminusI @ X
        inv = tf.linalg.inv(XtCiX + lambduhI)
        return tf.squeeze(inv @ tf.transpose(X) @ Cv @ pv)

        """
        return tf.linalg.inv(
            XtX + tf.transpose(X) @ (tf.linalg.diag(v) - tf.linalg.eye(tf.shape(v)[0])) @ X + lambduhI
            ) @ tf.transpose(X) @ tf.linalg.diag(v) @ tf.where(v > 0., 1., 0.)
        """

    @tf.function
    def update_items(self, C):
        XtX = tf.transpose(self.X) @ self.X
        lambduhI = tf.linalg.eye(self.latent_dim) * self.lambduh
        for i in tf.range(self.num_items):
            iv = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(C, [0, i], [self.num_users, 1])
                )
            )
            self.Y[i].assign(self.vector_update(XtX, self.X, iv, lambduhI))

    @tf.function
    def update_users(self, C):
        YtY = tf.transpose(self.Y) @ self.Y
        lambduhI = tf.linalg.eye(self.latent_dim) * self.lambduh
        for u in tf.range(self.num_users):
            uv = tf.squeeze(
                tf.sparse.to_dense(
                    tf.sparse.slice(C, [u, 0], [1, self.num_items])
                )
            )
            self.X[u].assign(self.vector_update(YtY, self.Y, uv, lambduhI))
