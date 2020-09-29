from recsyslib.als.alsmixin import ALSMixin
import tensorflow as tf


class BVD(ALSMixin):
    def __init__(
        self,
        num_users,
        num_items,
        num_rowclusters,
        num_columnclusters,
        name="bvd",
        **kwargs
    ):
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
        self.latent_dim2 = num_columnclusters
        super().__init__(
            num_users, num_items, latent_dim=num_rowclusters, **kwargs
        )
        self.R = tf.Variable(
            tf.random.uniform(
                minval=0.001,
                maxval=1.0,
                shape=(self.num_users, self.latent_dim),
            ),
            trainable=False,
            name="R",
        )
        self.B = tf.Variable(
            tf.random.uniform(
                minval=0.001,
                maxval=1.0,
                shape=(self.latent_dim, self.latent_dim2),
            ),
            trainable=False,
            name="B",
        )
        self.C = tf.Variable(
            tf.random.uniform(
                minval=0.001,
                maxval=1.0,
                shape=(self.latent_dim2, self.num_items),
            ),
            trainable=False,
            name="C",
        )

    def interact_to_confidence(self, y):
        return y

    @property
    def item_embeddings(self):
        return self.C.numpy().T

    @property
    def user_embeddings(self):
        return self.R.numpy()

    def update_R(self, Z, C, B, R):
        return (
            R
            * (
                tf.sparse.sparse_dense_matmul(Z, tf.transpose(C))
                @ tf.transpose(B)
            )
            / (R @ B @ C @ tf.transpose(C) @ tf.transpose(B))
        )

    def update_B(self, Z, C, B, R):
        return (
            B
            * (tf.transpose(R) @ tf.sparse.to_dense(Z) @ tf.transpose(C))
            / (tf.transpose(R) @ R @ B @ C @ tf.transpose(C))
        )

    def update_C(self, Z, C, B, R):
        return (
            C
            * (tf.transpose(B) @ tf.transpose(R) @ tf.sparse.to_dense(Z))
            / (tf.transpose(B) @ tf.transpose(R) @ R @ B @ C)
        )

    @tf.function
    def call(self, inputs):
        Z = inputs
        self.R.assign(self.update_R(Z, self.C, self.B, self.R))
        self.B.assign(self.update_B(Z, self.C, self.B, self.R))
        self.C.assign(self.update_C(Z, self.C, self.B, self.R))

    @tf.function
    def mse(self, Z):
        # eq (1)
        return tf.reduce_mean(
            tf.pow(tf.sparse.to_dense(Z) - self.R @ self.B @ self.C, 2)
        )
