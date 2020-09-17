import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class BurnIn(tf.keras.callbacks.Callback):
    def __init__(self, burnin_lr, burn_epochs=10):
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
        if epoch < self.burn_epochs:
            lr = self.burnin_lr
        else:
            lr = self.lr
        K.set_value(self.model.optimizer.lr, lr)
        self.model.logger.info(f"Epoch {epoch}'s lr is {lr}")


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


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(
                    "Restoring model weights from the end of the best epoch."
                )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.model.logger.info(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
