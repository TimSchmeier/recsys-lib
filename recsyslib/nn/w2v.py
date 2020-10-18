import tensorflow as tf
import tensorflow_recommenders as tfrs
from recsys.nn.mf import EntityModel
import logging


class W2V(EntityModel, tfrs.models.Model):
    def __init__(self, unique_ids, dim=32, **kwargs):
        """Implimentation of:
            Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space."
            arXiv:1301.3781
        Args:
            unique_item_ids (list[Int]): all unique symbols to embed.
            **kwargs
        """
        super().__init__(unique_ids, dim, **kwargs)
        self.logger = logging.getLogger()
        self.context = tf.keras.Sequential(
            [
                self.entity_lookup,
                tf.keras.layers.Embedding(
                    self.entity_lookup.vocab_size(), dim, name="IdContext"
                ),
            ]
        )

    def set_task(self, item_dataset):
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_dataset.batch(128).map(self.item_model)
            ),
        )

    @tf.function
    def call(self, inputs):
        w, c = inputs
        return self.entity_embedding(w), self.context(c)

    @tf.function
    def compute_loss(self, inputs, training=False):
        x, y, sample_weights = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        item_embeddings, context_embeddings = self(x)
        return self.task(item_embeddings, context_embeddings, sample_weights)
