import numpy as np
from get_data import DataGetter
from modelmixin import ModelMixin
from scipy.sparse import csr_matrix
from tqdm import tqdm


class BVD(ModelMixin):
    def __init__(self, num_users, num_items, name, **kwargs):
        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.R = np.random.normal(size=(self.num_items, self.latent_dim))
        self.B = np.random.normal(size=(self.latent_dim, self.latent_dim))
        self.C = np.random.normal(size=(self.num_users, self.latent_dim))

    @property
    def item_embeddings(self):
        return self.C

    @property
    def user_embeddings(self):
        return self.R

    def fit(self, x, y, epochs=20):
        users, items = x
        ratings = y
        Z = csr_matrix(
            (ratings, (users, items)), shape=(self.num_users, self.num_items)
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(Z)
            self.logging.info(f"epoch {it} complete")

    def call(self, inputs):
        Z = inputs
        # update R
        ZCtBt = self.Z @ self.C.T @ self.B.T
        RBCCtBt = self.R @ self.B @ self.C @ self.C.T @ self.B.T
        self.R = self.R @ (ZCtBt / RBCCtBt)
        # update B
        RtZCt = self.R.T @ self.Z @ self.C.T
        RtRBCCt = self.R.T @ self.R @ self.B @ self.C @ self.C.T
        self.B = self.B @ (RtZCt / RtRBCCt)
        # update C
        BtRtZ = self.B.T @ self.R.T @ Z
        BtRtRBC = self.B.T @ self.R.T @ self.R @ self.B @ self.C
        self.C = self.C @ (BtRtZ / BtRtRBC)

    def predict(self, user):
        return self.X[user, :].T @ self.Y


if __name__ == "__main__":
    d = DataGetter()
    df = d.get_ml_data()
    df = d.assign_indices(df)
    movieId_to_idx = d.get_item_idx_map(df)
    userId_to_idx = d.get_user_idx_map(df)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    bvd = BVD(num_users, num_movies, "imf")
    bvd.fit(
        (df["user_idx"].values, df["movie_idx"].values),
        df["rating"].values,
        epochs=1,
    )
