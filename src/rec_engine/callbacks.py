import tensorflow as tf
import numpy as np

from rec_engine import log


## Custom callbacks
# Custom Early Stopping
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    A custom early stopping callback for Keras models, which monitors both retrieval and rating metrics
    during training. The callback stops the training process if no improvements are observed in the specified
    metrics within the defined patience period, starting from a given epoch.

    Attributes
    ----------
    patience : int
        Number of epochs with no improvement after which training will be stopped. Default is 0.
    start_from_epoch : int
        The epoch number from which to start monitoring the metrics for early stopping. Default is 3.
    delta_retrieval : float
        Minimum change in the retrieval metric to qualify as an improvement. Default is 0.01.
    delta_rating : float
        Minimum change in the rating metric to qualify as an improvement. Default is 0.01.
    retrieval_metric : str, optional
        The name of the retrieval metric to monitor. Default is "val_factorized_top_k/top_50_categorical_accuracy".
    rating_metric : str, optional
        The name of the rating metric to monitor. Default is "val_root_mean_squared_error".
    verbose : bool
    """
    def __init__(
            self, patience=0, start_from_epoch=3,
            delta_retrieval=0.01, delta_rating=0.01,
            retrieval_metric: str = "val_factorized_top_k/top_50_categorical_accuracy",
            rating_metric: str = "val_root_mean_squared_error",
            verbose=True
        ):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.delta_retrieval = delta_retrieval
        self.delta_rating = delta_rating
        self.retrieval_metric = retrieval_metric
        self.rating_metric = rating_metric
        self.verbose = verbose
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None) -> None:
        self.wait = 0
        self.retrieval_metric_value = -np.Inf
        self.rating_metric_value = np.Inf

    def on_epoch_end(self, epoch, logs=None) -> None:
        current_retrieval_metric = logs.get(self.retrieval_metric)
        current_rating_metric = logs.get(self.rating_metric)
        improvement = False

        if epoch < self.start_from_epoch or current_retrieval_metric is None or current_rating_metric is None:
            return

        # Check for improvement in the retrieval metric
        if current_retrieval_metric is not None and np.greater(current_retrieval_metric - self.retrieval_metric_value, self.delta_retrieval):
            self.retrieval_metric_value = current_retrieval_metric
            self.wait = 0
            improvement = True

        # Check for improvement in the rating metric
        if current_rating_metric is not None and np.less(current_rating_metric - self.rating_metric_value, -self.delta_rating):
            self.rating_metric_value = current_rating_metric
            self.wait = 0
            improvement = True

        # If no improvement is observed, increase the wait counter and stop training if patience is exceeded
        if not improvement:
            self.wait += 1
            if self.wait >= self.patience and epoch >= self.start_from_epoch:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs = None) -> None:
        if self.stopped_epoch > 0 and self.verbose:
            log.info(f"Epoch {self.stopped_epoch + 1}: early stopping")


class CustomLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, lr_schedule) -> None:
        self.lr_schedule = lr_schedule

    def on_epoch_begin(self, epoch, logs=None) -> None:
        if epoch < len(self.lr_schedule):
            new_lr = self.lr_schedule[epoch]
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            log.info(f"\nEpoch {epoch+1}: Learning rate set to {new_lr}")