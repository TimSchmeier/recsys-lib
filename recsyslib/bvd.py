from modelmixin import ModelMixin
from tqdm import tqdm
import tensorflow as tf


class BVD(ModelMixin):
    def __init__(self, num_users, num_items, name="bvd", **kwargs):
        """Implimentation of:
            Long, B. Zhang, Z., Yu, P. 2005. Co-clustering by block value decomposition.
            KDD '05: Pages 635–640 https://doi.org/10.1145/1081870.1081949

        This implimentation uses tensorflow for faster compiled math operations, not optimization.
        These are multiplicative updates as in the original paper.

        Args:
            num_users (int): number of users
            num_items (int): number of items
            name (str): date/model name
        """

        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.R = tf.Variable(
            tf.random.normal(shape=(self.num_items, self.latent_dim)),
            trainable=False,
        )
        self.B = tf.Variable(
            tf.random.normal(shape=(self.latent_dim, self.latent_dim)),
            trainable=False,
        )
        self.C = tf.Variable(
            tf.random.normal(shape=(self.num_users, self.latent_dim)),
            trainable=False,
        )

    @property
    def item_embeddings(self):
        return self.C.numpy()

    @property
    def user_embeddings(self):
        return self.R.numpy()

    def fit(self, x, y, epochs=20):
        """Run block value decomposition on user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """
        users, items = x
        ratings = y
        indices = [[u, i] for u, i in zip(users, items)]
        Z = tf.sparse.reorder(
            tf.sparse.SparseTensor(
                indices, ratings, dense_shape=[self.num_users, self.num_items]
            )
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(Z)
            self.logging.info(f"epoch {it} complete")

    def update_R(self, Z):
        ZCtBt = tf.sparse.sparse_dense_matmul(
            Z, tf.transpose(self.C)
        ) @ tf.transpose(self.B)
        RBCCtBt = (
            self.R
            @ self.B
            @ self.C
            @ tf.transpose(self.C)
            @ tf.transpose(self.B)
        )
        self.R = self.R @ (ZCtBt / RBCCtBt)

    def update_B(self, Z):
        RtZCt = (
            tf.sparse.sparse_dense_matmul(tf.sparse.transpose(Z), self.R)
            @ Z
            @ tf.transpose(self.C)
        )
        RtRBCCt = (
            tf.transpose(self.R)
            @ self.R
            @ self.B
            @ self.C
            @ tf.transpose(self.C)
        )
        self.B = self.B @ (RtZCt / RtRBCCt)

    def update_C(self, Z):
        BtRtZ = tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(Z),
            tf.transpose(tf.transpose(self.B) @ tf.transpose(self.R)),
        )
        BtRtRBC = (
            tf.transpose(self.B)
            @ tf.transpose(self.R)
            @ self.R
            @ self.B
            @ self.C
        )
        self.C = self.C @ (BtRtZ / BtRtRBC)

    def call(self, inputs):
        Z = inputs
        self.update_R(Z)
        self.logger.info("updated R")
        self.update_B(Z)
        self.logger.info("updated B")
        self.update_C(Z)
        self.logger.info("updated C")
