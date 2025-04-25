# callbacks.py
"""Definitions of Keras callbacks for monitoring training."""

import os
import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import config # Import configuration

def get_callbacks(monitor_metric='val_weighted_weather_penalty', model_checkpoint=False, checkpoint_path='best_model.keras'):
    """
    Returns a list of standard Keras callbacks.

    Args:
        monitor_metric (str): The metric name to monitor for EarlyStopping and ReduceLROnPlateau.
        model_checkpoint (bool): Whether to include ModelCheckpoint callback.
        checkpoint_path (str): Filepath for saving the best model if model_checkpoint is True.

    Returns:
        list: A list of Keras Callback instances.
    """
    # Dynamically create the TensorBoard log path using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.LOG_DIR_BASE, timestamp)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    callbacks_list = [
        EarlyStopping(
            monitor=monitor_metric,
            patience=config.EARLY_STOPPING_PATIENCE, # Number of epochs with no improvement after which training will be stopped.
            restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity.
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=config.REDUCE_LR_FACTOR, # Factor by which the learning rate will be reduced. new_lr = lr * factor.
            patience=config.REDUCE_LR_PATIENCE, # Number of epochs with no improvement after which learning rate will be reduced.
            min_lr=config.MIN_LR, # Lower bound on the learning rate.
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,        # Directory where TensorBoard logs will be saved.
            histogram_freq=1,       # Frequency (in epochs) at which to compute activation and weight histograms.
            write_graph=True,       # Whether to visualize the graph in TensorBoard.
            write_images=False,     # Whether to write model weights to visualize as image in TensorBoard.
            update_freq='epoch',    # When to update the logs ('batch' or 'epoch').
            profile_batch=0         # Profile the batch to sample compute characteristics. 0 disables profiling.
        )
    ]

    if model_checkpoint:
        # Add ModelCheckpoint only if required
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path, # Use the provided path to save the model file.
            monitor=monitor_metric,   # Metric to monitor for saving the best model.
            save_best_only=True,      # Only save the model when the monitored quantity has improved.
            save_weights_only=False,  # If False, the full model is saved. If True, only the weights are saved.
            verbose=1
        )
        callbacks_list.append(checkpoint)
        print(f"ModelCheckpoint active. Best model will be saved to: {checkpoint_path}")

    return callbacks_list