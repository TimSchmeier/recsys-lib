import tensorflow as tf
from tensorflow.keras import layers
from recsyslib.modelmixin import ModelMixin


class MF(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, biases=True, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.use_biases = biases
        self.user_embedding = layers.Embedding(
            self.num_users,
            self.latent_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.item_embedding = layers.Embedding(
            self.num_items,
            self.latent_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        if self.use_biases:
            self.user_bias = layers.Embedding(num_users, 1)
            self.item_bias = layers.Embedding(num_items, 1)

    @property
    def item_embeddings(self):
        if self.use_biases:
            return self.item_embeddings + self.item_bias
        else:
            return self.item_embeddings

    @property
    def user_embeddings(self):
        if self.use_biases:
            return self.user_embeddings + self.user_bias
        else:
            return self.user_embeddings

    @tf.function
    def get_dot(self, inputs):
        """Forward pass.

        Args:
            inputs (tuple): (userIdx, itemIdx)

        Returns:
            tensor: dot product of user and item
        """
        user, item = inputs
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        dot_product = tf.reduce_sum(
            tf.multiply(user_vector, item_vector), axis=1
        )
        if self.use_biases:
            user_bias = self.user_bias(user)
            item_bias = self.item_bias(item)
            dot_product = dot_product + user_bias + item_bias
        return dot_product

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs (tuple): (userIdx, itemIdx)

        Returns:
            tensor: positive thresholded dot product of user and item
        """
        dot = self.get_dot(inputs)
        # ratings are only positive
        return tf.nn.relu(dot)


class LogisticMF(MF):
    def __init__(self, num_users, num_items, biases=True, **kwargs):
        super().__init__(num_users, num_items, biases=True, **kwargs)

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs (tuple): (userIdx, itemIdx)

        Returns:
            tensor: logistic(dot product of user and item)
        """
        dot_product = self.get_dot(inputs)
        logistic = dot_product / (1.0 + dot_product)
        return logistic
