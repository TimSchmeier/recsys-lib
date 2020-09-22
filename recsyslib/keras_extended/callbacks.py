import tensorflow as tf
import tensorflow.keras.backend as K


class BurnIn(tf.keras.callbacks.Callback):
    def __init__(self, burnin_lr=None, burn_epochs=10):
        super().__init__()
        self.burn_epochs = burn_epochs
        self.burnin_lr = burnin_lr
        self.lr = None

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer must have lr")
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # set learning rate from optimizer arg
        if epoch == 0:
            self.lr = lr
            if not self.burnin_lr:
                self.burnin_lr = self.lr / 10.0
        if epoch < self.burn_epochs:
            lr = self.burnin_lr
            self.model.logger.info(f"Burn In Epoch {epoch}'s lr is {lr}")
        else:
            lr = self.lr
        K.set_value(self.model.optimizer.lr, lr)


class AnnealKLloss(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = tf.Variable(
            num_epochs, dtype=tf.float32, trainable=False
        )

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "kl_weight"):
            raise ValueError("model must have kl_weight")

        # linearly increase kl divergence loss
        epoch_kl_weight = (epoch + 1) / self.num_epochs
        K.set_value(self.model.kl_weight, epoch_kl_weight)
        self.model.logger.info(
            f"Epoch {epoch}'s kl_weight is {epoch_kl_weight}"
        )


class DecayLR(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, lr=None, decay_rate=0.96, decay_steps=20):
        super().__init__()
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def decay(self, epoch):
        return self.lr * self.decay_rate ** (epoch / self.decay_steps)

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        if epoch == 0 and self.lr is None:
            self.lr = lr
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.decay(epoch)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        self.model.logger.info(
            f"Epoch {epoch}: Learning rate is {scheduled_lr}."
        )
