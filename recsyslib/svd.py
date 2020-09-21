from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from recsyslib.modelmixin import ModelMixin
import numpy as np


class SVD(ModelMixin):
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

    @property
    def user_embeddings(self):
        return self.U * np.sqrt(self.S.reshape(1, -1))

    @property
    def item_embeddings(self):
        return (np.sqrt(self.S.reshape(-1, 1)) * self.Vt).T

    def fit(self, x, y):
        """Run singular value decomposition on a user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """

        users, items = x
        ratings = y
        user_by_item = csr_matrix(
            (ratings, (users, items)), shape=(self.num_users, self.num_items)
        )
        self.call(user_by_item)

    def call(self, inputs):
        self.U, self.S, self.Vt = svds(inputs, k=self.latent_dim)
        self.logger.info("fit complete")
