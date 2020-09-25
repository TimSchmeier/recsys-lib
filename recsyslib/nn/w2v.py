import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from recsyslib.modelmixin import ModelMixin


class W2V(ModelMixin, tf.keras.Model):
    def __init__(self, num_items, **kwargs):
        """Implimentation of:
            Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space".
            arXiv:1301.3781

        Args:
            num_items (int): total number of sequence items to embed.
            **kwargs
        """
        super().__init__(num_users=None, num_items=num_items, **kwargs)
        self.hidden = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )
        self.context = tf.keras.layers.Embedding(
            self.num_items, self.latent_dim
        )

    @tf.function
    def call(self, inputs):
        """Embed word pairs.

        Args:
            inputs (tuple): (word, context)
        """
        word, context = inputs
        w = self.hidden(word)
        c = self.context(context)
        dot_product = tf.keras.layers.dot([w, c], axes=(1))
        return tf.nn.sigmoid(dot_product)

    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            ypred = self.call(x)
            loss = tf.reduce_mean(binary_crossentropy(y, ypred))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, ypred)
        return {m.name: m.result() for m in self.metrics}
