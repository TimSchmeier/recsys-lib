import numpy as np
from recsyslib.modelmixin import ModelMixin
from tqdm import tqdm
import tensorflow as tf


class LogisticALS(ModelMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="logistic",
        alpha=40,
        lambduh=100,
        gamma=1.0,
        **kwargs,
    ):
        """Implimentation of:
            Christopher C. Johnson. 2014. Logistic matrix factorization for implicit feedback data.
            In NIPS 2014 Workshop on Distributed Machine Learning and Matrix Computations.

        This implimentation uses tensorflow for faster compiled math operations, not optimization.
        These are multiplicative updates as in the original paper.

        Args:
            num_users (int): number of users
            num_items (int): number of items
            name (str): date/model name
            alpha (float): used to compute confidence value
            lambduh (float): L2 regularization strength
            gamma (float): hyperparameter for Adagrad updates
        """

        self.name = name
        super().__init__(num_users, num_items, **kwargs)
        self.alpha = alpha
        self.lambduh = lambduh
        self.gamma = gamma
        self.EPS = 1e-5
        self.Y = tf.Variable(
            tf.random.normal(shape=(self.num_items, self.latent_dim)),
            trainable=False,
        )
        self.X = tf.Variable(
            tf.random.normal(shape=(self.num_users, self.latent_dim)),
            trainable=False,
        )
        self.Bu = tf.Variable(
            tf.random.normal(shape=(self.num_users, 1)),
            trainable=False,
        )
        self.Bi = tf.Variable(
            tf.random.normal(shape=(1, self.num_items)), trainable=False
        )
        # Adagrad updates
        self.sum_sq_grad_Y = tf.Variable(
            tf.zeros_like(self.Y), trainable=False
        )
        self.sum_sq_grad_X = tf.Variable(
            tf.zeros_like(self.X), trainable=False
        )
        self.sum_sq_grad_Bu = tf.Variable(
            tf.zeros_like(self.Bu), trainable=False
        )
        self.sum_sq_grad_Bi = tf.Variable(
            tf.zeros_like(self.Bi), trainable=False
        )

    @property
    def item_embeddings(self):
        return (self.Y + tf.transpose(self.Bi)).numpy()

    @property
    def user_embeddings(self):
        return (self.X + self.Bu).numpy()

    def fit(self, x, y, epochs=20):
        """Run Implicit ALS on user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """
        users, items = x
        ratings = self.interact_to_confidence(y).astype(np.float32)
        indices = [[u, i] for u, i in zip(users, items)]
        alphaR = tf.sparse.reorder(
            tf.sparse.SparseTensor(
                indices=indices,
                values=ratings,
                dense_shape=[self.num_users, self.num_items],
            )
        )
        self.logger.info("begin training")
        for it in tqdm(range(epochs)):
            self.call(alphaR)
            self.logger.info(f"epoch {it} complete")

    def _clip_grad(self, grad, val=80.0):
        # protect against exp(grad) overflows
        return tf.clip_by_value(grad, -10, 10)

    def loginteract_to_confidence(self, val):
        # eq (1.5)
        return 1.0 + self.alpha * np.log(1.0 + val / self.EPS)

    def interact_to_confidence(self, val):
        return self.alpha * val

    @tf.function
    def get_common_grad(self, alphaR):
        grad = tf.math.exp(
            self._clip_grad(
                (self.X @ tf.transpose(self.Y) + self.Bu + self.Bi)
            )
        )
        grad /= tf.constant(1.0) + grad
        grad *= tf.constant(1.0) + tf.sparse.to_dense(alphaR)
        return grad

    @tf.function
    def update_users(self, alphaR):
        # eq (5)
        grad = self.get_common_grad(alphaR)
        gradX = (
            tf.sparse.sparse_dense_matmul(alphaR, self.Y)
            - grad @ self.Y
            - self.lambduh * self.X
        )
        gradBu = tf.sparse.reduce_sum(
            alphaR, axis=1, keepdims=True
        ) - tf.reduce_sum(grad, axis=1, keepdims=True)
        # eq (7)
        self.sum_sq_grad_X.assign_add(tf.math.square(gradX))
        self.X.assign_add(
            (self.gamma * gradX) / tf.math.sqrt(self.sum_sq_grad_X)
        )
        self.sum_sq_grad_Bu.assign_add(tf.math.square(gradBu))
        self.Bu.assign_add(
            (self.gamma * gradBu) / tf.math.sqrt(self.sum_sq_grad_Bu)
        )
        tf.debugging.check_numerics(self.X, "X not numeric")
        tf.debugging.check_numerics(self.Bu, "Bu not numeric")

    @tf.function
    def update_items(self, alphaR):
        grad = self.get_common_grad(alphaR)
        gradY = (
            tf.sparse.sparse_dense_matmul(tf.sparse.transpose(alphaR), self.X)
            - tf.transpose(grad) @ self.X
            - self.lambduh * self.Y
        )
        gradBi = tf.sparse.reduce_sum(alphaR, axis=0) - tf.reduce_sum(
            grad, axis=0, keepdims=True
        )
        self.sum_sq_grad_Y.assign_add(tf.math.square(gradY))
        self.Y.assign_add(
            (self.gamma * gradY) / tf.math.sqrt(self.sum_sq_grad_Y)
        )
        self.sum_sq_grad_Bi.assign_add(tf.math.square(gradBi))
        self.Bi.assign_add(
            (self.gamma * gradBi) / tf.math.sqrt(self.sum_sq_grad_Bi)
        )
        tf.debugging.check_numerics(self.Y, "Y not numeric")
        tf.debugging.check_numerics(self.Bi, "Bi not numeric")

    def call(self, inputs):
        """Run ALS.

        Args:
            inputs (csr_matrix): confidence matrix
        """
        self.update_items(inputs)
        self.logger.info("updated items")
        self.update_users(inputs)
        self.logger.info("updated users")
