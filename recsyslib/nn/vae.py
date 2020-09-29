import tensorflow as tf
from tensorflow.keras import layers
from recsyslib.modelmixin import ModelMixin


class Sampling(layers.Layer):
    def __init__(self):
        super().__init__(trainable=False, name="sampling")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEncoder(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, num_dense, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = []
        for layer in range(num_dense):
            self.dense_layers.append(
                layers.Dense(
                    600,
                    activation="tanh",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
        self.dense_mu = layers.Dense(self.latent_dim, name="z_mean")
        self.dense_var = layers.Dense(self.latent_dim, name="z_log_var")
        self.sample = Sampling()

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        z_mean, z_log_var = self.dense_mu(x), self.dense_var(x)
        z = self.sample((z_mean, z_log_var))
        return z_mean, z_log_var, z


class VADecoder(ModelMixin, tf.keras.Model):
    def __init__(self, num_users, num_items, num_dense, **kwargs):
        super().__init__(num_users, num_items, **kwargs)
        self.dense_layers = []
        for layer in range(num_dense):
            self.dense_layers.append(
                layers.Dense(
                    600,
                    activation="tanh",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(
                        mean=0.0, stddev=0.05
                    ),
                )
            )
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


class VAE(ModelMixin, tf.keras.Model):
    def __init__(
        self,
        num_users,
        num_items,
        num_dense=2,
        latent_dim=200,
        kl_weight=tf.Variable(0.0, dtype=tf.float32, trainable=False),
        **kwargs
    ):
        """Implimentation of:
            Liang, D. et.  al. 2018. Variational Autoencoders for Collaborative Filtering.
             WWW '18. Pages 689–698 https://doi.org/10.1145/3178876.3186150

        Args:
            num_users (int): number of users
            num_items (int): number of items
            num_dense (int): number of dense layers in each of the encoder and decoder.
            **kwargs
        """
        super().__init__(num_users, num_items, **kwargs)
        self.kl_weight = kl_weight
        self.enc = VAEncoder(num_users, num_items, num_dense, name="enc")
        self.dec = VADecoder(num_users, num_items, num_dense, name="dec")

    def call(self, inputs):
        z_mean, z_log_var, z = self.enc(inputs)
        reconstructed_x = self.dec(z)
        return z_mean, z_log_var, z, reconstructed_x

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstructed_x = self.call(inputs)
            reconstruction_loss = (
                tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        inputs, reconstructed_x
                    )
                )
                * self.num_items
            )
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss_weighted = kl_loss * self.kl_weight  # annealing
            total_loss = reconstruction_loss + kl_loss_weighted
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "kl_weight": self.kl_weight,
            "kl_loss_weighted": kl_loss_weighted,
        }
