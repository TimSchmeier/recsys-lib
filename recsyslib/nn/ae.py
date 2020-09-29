import tensorflow as tf
from tensorflow.keras import layers
from recsyslib.modelmixin import ModelMixin


class Encoder(ModelMixin, tf.keras.Model):
    def __init__(
        self, num_users, num_items, num_dense=1, dense_units=600, **kwargs
    ):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = [
            layers.Dense(
                dense_units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0.0, stddev=0.05
                ),
            )
            for _ in range(num_dense)
        ]
        self.dense_out = layers.Dense(self.latent_dim)

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        encoded = self.dense_out(x)
        return encoded


class Decoder(ModelMixin, tf.keras.Model):
    def __init__(
        self, num_users, num_items, num_dense=1, dense_units=600, **kwargs
    ):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = [
            layers.Dense(
                dense_units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=0.0, stddev=0.05
                ),
            )
            for _ in range(num_dense)
        ]
        self.dense_out = layers.Dense(
            self.num_items, activation="sigmoid", name="out"
        )

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        recon_x = self.dense_out(x)
        return recon_x


class AE(ModelMixin, tf.keras.Model):
    def __init__(
        self,
        num_users,
        num_items,
        num_dense=1,
        dense_units=600,
        latent_dim=200,
        **kwargs
    ):
        super().__init__(num_users, num_items, **kwargs)
        self.enc = Encoder(
            num_users, num_items, num_dense, dense_units, name="enc"
        )
        self.dec = Decoder(
            num_users, num_items, num_dense, dense_units, name="dec"
        )

    @tf.function
    def call(self, inputs):
        h = self.enc(inputs)
        return self.dec(h)

    def train_step(self, inputs):
        inputs = tf.sparse.to_dense(inputs)
        with tf.GradientTape() as tape:
            recon_x = self(inputs)
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(inputs, recon_x)
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        return {"loss": loss}

    def fit(self, x, y, **kwargs):
        M = self.build_sparse_matrix(x, y)
        super().fit(x=M, **kwargs)
