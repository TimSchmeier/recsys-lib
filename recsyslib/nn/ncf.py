import tensorflow as tf
from tensorflow.keras import layers
from recsyslib.mf import MatrixFactorization


class NeuMF(MatrixFactorization):
    def __init__(self, num_users, num_items, num_dense=3, **kwargs):
        """
        Implimentation of:
            He, X. et. al. 2017. Neural Collaborative Filtering.
            WWW '17: Pages 173–182 https://doi.org/10.1145/3038912.3052569

        Args:
            num_users (int): number of users
            num_items (int): number of items
            num_dense (int): depth of MLP
            **kwargs

        """
        super().__init__(num_users, num_items, biases=False, **kwargs)
        self.num_dense = num_dense
        self.dense_layers = [
            layers.Dense(
                (self.latent_dim * 2) / (layer ** 2), activation="relu"
            )
            for layer in range(1, num_dense)
        ]
        self.dropout_layers = [
            layers.Dropout(rate=0.4) for _ in range(num_dense)
        ]
        self.output_layer = layers.Dense(1)

    @tf.function
    def call(self, inputs):
        user, item = inputs
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        x = tf.concat([user_vector, item_vector], axis=1)
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)
        # Figure 3 / eq (11)
        user_item_product = user_vector * item_vector
        to_output = tf.concat([user_item_product, x], axis=1)
        logit = self.output_layer(to_output)
        return tf.nn.sigmoid(logit)

    def train_step(self, inputs):
        X, y = inputs
        with tf.GradientTape() as tape:
            yhat = self(X)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, yhat))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        return {"loss": loss}
