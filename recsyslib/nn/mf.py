import tensorflow as tf
import tensorflow_recommenders as tfrs
import logging


class EntityModel(tf.keras.Model):
    def __init__(self, unique_ids, dim=32, **kwargs):
        """Generic model for embedding items with an Integer Id.
        Args:
            unique_ids (list[Int]): unique symbols to embed."""
        super().__init__(**kwargs)
        self.num_entities = len(unique_ids)
        self.dim = dim
        self.entity_lookup = (
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=unique_ids, name="IdLookup"
            )
        )
        self.entity_embedding = tf.keras.Sequential(
            [
                self.entity_lookup,
                tf.keras.layers.Embedding(
                    self.entity_lookup.vocab_size(), dim, name="IdEmbedding"
                ),
            ]
        )

    @tf.function
    def call(self, inputs):
        return self.entity_embedding(inputs)


class RetrievalMF(tfrs.models.Model):
    def __init__(self, unique_user_ids, unique_item_ids, **kwargs):
        """Retrieval Model, minimizes the cross entropy between item and user embeddings.
        Uses ratings as sample weights if applicable.
        Args:
            unique_user_ids (list[Int]): user symbols to embed.
            unique_item_ids (list[Int]): item symbols to embed.
        """
        super().__init__(**kwargs)
        self.user_model = EntityModel(unique_user_ids)
        self.item_model = EntityModel(unique_item_ids)
        self.logger = logging.getLogger()

    def set_task(self, item_dataset):
        """
        Args:
            item_dataset: tf.data.Dataset of item ids
        """
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_dataset.batch(128).map(self.item_model),
            ),
        )

    @tf.function
    def call(self, inputs):
        user_ids, item_ids = inputs
        return self.user_model(user_ids), self.item_model(item_ids)

    @tf.function
    def compute_loss(self, inputs, training=False):
        x, y, sample_weights = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        user_embeddings, item_embeddings = self(x)
        return self.task(user_embeddings, item_embeddings, sample_weights)


class RankingMF(RetrievalMF):
    def __init__(self, unique_user_ids, unique_item_ids, **kwargs):
        """Implimentation of:
        He, X. et. al. 2017. Neural Collaborative Filtering.
        WWW '17: Pages 173–182 https://doi.org/10.1145/3038912.3052569
        Args:
            unique_user_ids (list[Int]): user symbols to embed.
            unique_item_ids (list[Int]): item symbols to embed.
        """
        super().__init__(unique_user_ids, unique_item_ids)
        self.rank = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        self.set_task()

    def set_task(self):
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    @tf.function
    def compute_loss(self, inputs, training=False):
        x, y, sample_weights = tf.keras.utils.unpack_x_y_sample_weight(inputs)
        predictions = self.rank(tf.concat([*self(x)], axis=1))
        return self.task(labels=y, predictions=predictions)


class LogisticRetrieval(tfrs.tasks.Retrieval):
    def __init__(
        self,
        loss=None,
        metrics=None,
        temperature=None,
        num_hard_negatives=None,
        name="LogisticRetrieval",
    ):
        super().__init__(loss, metrics, temperature, num_hard_negatives, name)

        self.__loss = (
            loss
            if loss
            else tf.keras.losses.CategoricalCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.SUM
            )
        )

    def _loss(self, y_true, y_pred, sample_weight):
        return self.__loss(
            y_true, y_pred / (1.0 + y_pred), sample_weight=sample_weight
        )


class LogisticMF(RetrievalMF):
    def __init__(self, unique_user_ids, unique_item_ids, **kwargs):
        """
        Args:
            unique_user_ids (list[Int]): user symbols to embed.
            unique_item_ids (list[Int]): item symbols to embed.
        """
        super().__init__(unique_user_ids, unique_item_ids, **kwargs)

    def set_task(self, item_dataset):
        self.task = LogisticRetrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_dataset.batch(128).map(self.item_model),
            ),
        )
