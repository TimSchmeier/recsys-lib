from recsyslib.als.alsmixin import ALSMixin
import tensorflow as tf


class SVD(ALSMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="svd",
        **kwargs,
    ):
        self.name = name
        super().__init__(
            num_users,
            num_items,
            **kwargs,
        )
        self.U, self.S, self.Vt = None, None, None

    # TODO store only latent_dim number of singular vectors/values
    @property
    def user_embeddings(self):
        return (self.U * tf.math.sqrt(self.S.reshape(1, -1))).numpy()

    @property
    def item_embeddings(self):
        return (
            tf.transpose(tf.math.sqrt(self.S.reshape(-1, 1)) * self.Vt)
        ).numpy()

    def interact_to_confidence(self, y):
        return y

    def fit(self, x, y):
        """Run singular value decomposition on a user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """

        user_by_item = self.build_sparse_matrix(x, y)
        self.call(tf.sparse.to_dense(user_by_item))

    def call(self, inputs):
        self.logger.info("fit begin")
        self.S, self.U, self.Vt = tf.linalg.svd(inputs)
        self.logger.info("fit complete")
