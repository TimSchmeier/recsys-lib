import os
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from get_data import DataGetter
from modelmixin import ModelMixin


class SVD(ModelMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name,
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
        return self.U * self.S.reshape(1, -1)

    @property
    def item_embeddings(self):
        return (self.S.reshape(-1, 1) * self.Vt).T

    def fit(self, x, y):
        users, items = x
        ratings = y
        user_by_item = csr_matrix(
            (ratings, (users, items)), shape=(self.num_users, self.num_items)
        )
        self.call(user_by_item)

    def call(self, inputs):
        self.U, self.S, self.Vt = svds(inputs, k=self.latent_dim)
        self.logger.info("fit complete")

    def transform(self):
        # reconstruct input matrix
        return self.item_embeddings @ self.item_embeddings

    def predict(self, user):
        return self.user_embeddings[user, :] @ self.item_embeddings


if __name__ == "__main__":
    d = DataGetter()
    df = d.get_ml_data()
    df = d.assign_indices(df)
    movieId_to_idx = d.get_item_idx_map(df)
    userId_to_idx = d.get_user_idx_map(df)
    movie_idx_to_title = d.get_item_idx_to_title(df)
    title_to_movie_idx = {v: k for k, v in movie_idx_to_title.items()}
    X_train, X_test, y_train, y_test = d.get_training_data(df)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    svd = SVD(num_users, num_movies, name="svd")

    svd.fit(
        (df["user_idx"].values, df["movie_idx"].values),
        df["rating"].values,
    )

    item_nns = build_index(svd.item_embeddings)
    user_nns = build_index(svd.user_embeddings)

    search = NNS()
    search.add_index(item_nns, title_to_movie_idx, "movie")
    search.add_index(user_nns, userId_to_idx, "user")

    uid = 51900
    pp.pprint(sorted(df[df["userId"] == uid]["title"].tolist()))
    pp.pprint(sorted(search.get_similar("user", uid, 10, "movie")))

    pp.pprint(
        sorted(
            search.get_similar(
                "movie",
                "Twilight Saga: Breaking Dawn - Part 1, The (2011)",
                10,
                "movie",
            )
        )
    )
    pp.pprint(
        sorted(
            search.get_similar(
                "movie", "The Hunger Games: Catching Fire (2013)", 10, "movie"
            )
        )
    )
    pp.pprint(
        sorted(
            search.get_similar(
                "movie", "Gone with the Wind (1939)", 10, "movie"
            )
        )
    )
