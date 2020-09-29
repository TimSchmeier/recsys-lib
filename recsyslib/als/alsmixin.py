from recsyslib.modelmixin import ModelMixin
from recsyslib.target_transforms import interact_to_confidence
from collections import namedtuple
import tensorflow as tf
from tqdm import tqdm


# mock keras history class for unified plotting
history = namedtuple("history", "history")


class ALSMixin(ModelMixin):
    def __init__(self, num_users, num_items, log=False, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.history = history([])
        self.log = log

    def transform_y(self, y):
        return interact_to_confidence(y)

    def fit(self, x, y, epochs=20):
        """Run ALS on user item interactions.

        Args:
            x (tuple): (user_index, item_index)
            y (float): measure of a user's preference for an item
        """
        M = self.build_sparse_matrix(x, y)
        self.logger.info("begin training")
        for it in tqdm(tf.range(epochs)):
            self.call(M)
            self.logger.info(f"epoch {it} complete")
            if it % 10 == 0:
                mse = self.mse(M)
                self.logger.info(f"epoch {it} MSE: {mse}")
                self.history.history.append(mse.numpy())
        return self.history

    def mse(self, M):
        raise NotImplementedError("override me!")
