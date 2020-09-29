from recsyslib.als.alsmixin import ALSMixin
import numpy as np
from scipy.sparse import csr_matrix, linalg


class SVD(ALSMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="svd",
        **kwargs,
    ):
        """
        Vanilla SVD on a sparse matrix.

        Args:
            num_users (int): number of users
            num_items (int): number of items
            name (str): date/model name
        """

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
        return self.U * np.sqrt(self.S.reshape(1, -1))

    @property
    def item_embeddings(self):
        return (np.sqrt(self.S.reshape(-1, 1)) * self.Vt).T

    def interact_to_confidence(self, y):
        return y

    def build_sparse_matrix(self, x, y):
        return csr_matrix((y, x), shape=(self.num_users, self.num_items))

    def fit(self, x, y):
        """Run singular value decomposition on a user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """

        user_by_item = self.build_sparse_matrix(x, y)
        self.call(user_by_item)

    def call(self, inputs):
        self.logger.info("fit begin")
        self.U, self.S, self.Vt = linalg.svds(inputs, k=self.latent_dim)
        self.logger.info("fit complete")
