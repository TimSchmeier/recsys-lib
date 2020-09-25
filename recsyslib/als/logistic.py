from recsyslib.als.alsmixin import ALSMixin
import tensorflow as tf


class LogisticMF(ALSMixin):
    def __init__(
        self,
        num_users,
        num_items,
        name="logistic",
        alpha=40,
        lambduh=100,
        gamma=1.0,
        log=False,
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
            log (bool): if True, confidence computed as 1.0 + alpha * log(1 + interact); else alpha * interact
        """

        self.name = name
        super().__init__(num_users, num_items, log, **kwargs)
        self.alpha = alpha
        self.lambduh = lambduh
        self.gamma = gamma
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

    def _clip_grad(self, grad):
        # protect against exp(grad) overflows
        return tf.clip_by_value(grad, -10, 10)

    @tf.function
    def _get_common_grad(self, alphaR):
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
        grad = self._get_common_grad(alphaR)
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

    @tf.function
    def update_items(self, alphaR):
        grad = self._get_common_grad(alphaR)
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

    def call(self, inputs):
        """Run ALS.

        Args:
            inputs (csr_matrix): confidence matrix
        """
        self.update_items(inputs)
        self.logger.info("updated items")
        self.update_users(inputs)
        self.logger.info("updated users")

    @tf.function
    def mse(self, M):
        return tf.reduce_mean(
            tf.pow(
                M
                - (self.X @ tf.transpose(self.Y) + self.Bu + self.Bi)
                * self.alpha,
                2,
            )
        )
