import logging
import sys
import numpy as np
import tensorflow as tf


LATENT_DIM = 50


class ModelMixin:
    def __init__(
        self,
        num_users,
        num_items,
        latent_dim=LATENT_DIM,
        logfile=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            if logfile:
                fhandler = logging.FileHandler(logfile)
                fhandler.setLevel(logging.INFO)
                fhandler.setFormatter(formatter)
                self.logger.addHandler(fhandler)

    def transform_y(self, y):
        return y

    def build_sparse_matrix(self, x, y):
        users, items = x
        confidences = self.transform_y(y).astype(np.float32)
        indices = [[u, i] for u, i in zip(users, items)]
        M = tf.sparse.reorder(
            tf.sparse.SparseTensor(
                indices,
                confidences,
                dense_shape=[self.num_users, self.num_items],
            )
        )
        return M
