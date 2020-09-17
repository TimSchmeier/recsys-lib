import numpy as np
from get_data import DataGetter
from modelmixin import ModelMixin
from scipy.sparse import csr_matrix
from tqdm import tqdm


class LogisticMF(ModelMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name,
        alpha=40,
        lambduh=1e-6,
        gamma=1.0,
        precision=np.float16,
        **kwargs,
    ):
        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.alpha = alpha
        self.lambduh = lambduh
        self.gamma = gamma
        self.EPS = 1e-5
        self.Y = np.random.normal(
            size=(self.num_items, self.latent_dim)
        ).astype(precision)
        self.X = np.random.normal(
            size=(self.num_users, self.latent_dim)
        ).astype(precision)
        self.Bu = np.random.normal(size=(self.num_users, 1)).astype(precision)
        self.Bi = np.random.normal(size=(1, self.num_items)).astype(precision)
        # Adagrad updates
        self.sum_sq_grad_Y = np.zeros_like(self.Y)
        self.sum_sq_grad_X = np.zeros_like(self.X)
        self.sum_sq_grad_Bu = np.zeros_like(self.Bu)
        self.sum_sq_grad_Bi = np.zeros_like(self.Bi)

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
        ratings = self.interact_to_confidence(y)
        alphaR = csr_matrix(
            (ratings, (users, items)), shape=(self.num_users, self.num_items)
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(alphaR)
            self.logging.info(f"epoch {it} complete")

    def interact_to_confidence(self, val):
        # eq (1.5)
        return 1.0 + self.alpha * np.log(1.0 + val / self.EPS)

    def get_common_grad(self, alphaR):
        grad = np.exp(self.X @ self.Y.T + self.Bu + self.Bi)
        grad /= 1 + grad
        grad *= 1 + alphaR.todense()
        return grad

    def update_users(self, alphaR):
        # eq (5)
        grad = self.get_common_grad(alphaR)
        print("alphaR @ Y")
        gradX = alphaR @ self.Y
        print("grad @ Y")
        gradX -= grad @ self.Y
        print("lambda * X")
        gradX -= self.lambduh * self.X
        gradBu = (alphaR.sum(axis=1) - grad.sum(axis=1)).reshape(-1, 1)
        # eq (7)
        self.sum_sq_grad_X += gradX ** 2
        self.X += (self.gamma * gradX) / np.sqrt(self.sum_sq_grad_X)
        self.sum_sq_grad_Bu += gradBu ** 2
        self.Bu += (self.gamma * gradBu) / np.sqrt(self.sum_sq_grad_Bu)

    def update_items(self, alphaR):
        grad = self.get_common_grad(alphaR)
        print("alphaR, X, X, grad, Y")
        print(
            alphaR.shape, self.X.shape, self.X.shape, grad.shape, self.Y.shape
        )
        gradY = (
            (alphaR.T @ self.X) - (self.X @ grad.T) - (self.lambduh * self.Y)
        )
        gradBi = (alphaR.sum(axis=0) - grad.sum(axis=0)).reshape(1, -1)
        self.sum_sq_grad_Y += gradY ** 2
        self.Y += (self.gamma * gradY) / np.sqrt(self.sum_sq_grad_Y)
        self.sum_sq_grad_Bi += gradBi ** 2
        self.Bi += (self.gamma * gradBi) / np.sqrt(self.sum_sq_grad_Bi)

    def call(self, inputs):
        self.update_users(inputs)
        self.update_items(inputs)

    def predict(self, user):
        return self.X[user, :] @ self.Y.T


if __name__ == "__main__":
    d = DataGetter()
    movieId_to_idx = d.get_item_idx_map()
    userId_to_idx = d.get_user_idx_map()
    X, y = d.get_user_item_rating_tuples(split=False)

    num_users = len(userId_to_idx)
    num_movies = len(movieId_to_idx)

    lmf = LogisticMF(num_users, num_movies, latent_dim=2, name="logmf")
    lmf.fit(X, y.astype(np.float16), epochs=1)
