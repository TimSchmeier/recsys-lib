import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_data import DataGetter
from modelmixin import ModelMixin
from scipy.sparse import csr_matrix, eye, diags
from tqdm import tqdm


class ImplicitALS(ModelMixin):
    def __init__(
        self, num_users, num_items, name, alpha=40, lambduh=1e-6, **kwargs
    ):
        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.EPS = 1e-6
        self.alpha = alpha
        self.lambduh = lambduh
        self.Y = np.random.normal(size=(self.num_items, self.latent_dim))
        self.X = np.random.normal(size=(self.num_users, self.latent_dim))

    @property
    def item_embeddings(self, normalize=False):
        if normalize:
            return self.Y / np.linalg.norm(self.Y, axis=1)
        else:
            return self.Y

    @property
    def user_embeddings(self, normalize=False):
        if normalize:
            return self.X / np.linalg.norm(self.X, axis=1)
        else:
            return self.X

    def fit(self, x, y, epochs=20):
        users, items = x
        ratings = y  # maybe intercept here and change obs to confidences
        R_ui = csr_matrix(
            (ratings, (users, items)), shape=(self.num_users, self.num_items)
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(R_ui)
            self.logging.info(f"epoch {it} complete")

    def call(self, inputs):
        R_ui = inputs

        def C(r_ui):
            # eq (6)
            return 1.0 + self.alpha * np.log(1 + r_ui / self.EPS)

        def p(r_ui):
            # p_ui := 1 iff r_ui > 0
            p = np.zeros(r_ui.shape)
            p[np.argwhere(r_ui > 0)] = 1
            return p

        XtX = (
            self.X.T @ self.X
        )  # maybe make two functions, update_users and update_items
        I = eye(self.num_users)
        lambduhI = eye(self.latent_dim) * self.lambduh
        for i in range(self.num_items):
            iv = np.asarray(R_ui[:, i].todense()).squeeze()
            Ci = diags(C(iv))
            pi = p(iv)
            XtCiX = XtX + self.X.T @ (Ci - I) @ self.X
            inv = np.linalg.inv(XtCiX + lambduhI)
            self.Y[i] = inv @ self.X.T @ Ci @ pi
        YtY = self.Y.T @ self.Y
        I = eye(self.num_items)
        lambduhI = eye(self.latent_dim) * self.lambduh
        for u in range(self.num_users):
            uv = np.asarray(R_ui[u, :].todense()).squeeze()
            Cu = diags(C(uv))
            pu = p(uv)
            YtCuY = YtY + self.Y.T @ (Cu - I) @ self.Y
            inv = np.linalg.inv(YtCuY + lambduhI)
            self.X[u] = inv @ self.Y.T @ Cu @ pu

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

    imf = ImplicitALS(num_users, num_movies, "imf")
    imf.fit(
        (df["user_idx"].values, df["movie_idx"].values),
        df["rating"].values,
        epochs=1,
    )
