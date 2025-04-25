# metrics.py
"""Custom Keras metrics."""

import tensorflow as tf

class WeightedWeatherPenalty(tf.keras.metrics.Metric):
    """
    Custom weighted penalty metric for weather classification.

    The penalty is higher for misclassifying 'clear' (0) and 'rainy' (2) classes,
    and lower for misclassifying the 'cloudy' (1) class.
    A lower metric value indicates a better result.
    """
    def __init__(self, clear_idx=0, cloudy_idx=1, rainy_idx=2,
                 critical_penalty=10.0, non_critical_penalty=1.0,
                 name='weighted_weather_penalty', dtype=tf.float32):
        """
        Initializes the metric.

        Args:
            clear_idx (int): Index of the 'clear' class (critical). Defaults to 0.
            cloudy_idx (int): Index of the 'cloudy' class (non-critical). Defaults to 1.
            rainy_idx (int): Index of the 'rainy' class (critical). Defaults to 2.
            critical_penalty (float): Penalty for misclassifying a critical class.
            non_critical_penalty (float): Penalty for misclassifying a non-critical class.
            name (str): Name of the metric.
            dtype: Data type for metric calculations.
        """
        super().__init__(name=name, dtype=dtype)
        # Store indices as tensors for consistent TF operations
        self.clear_idx = tf.constant(clear_idx, dtype=tf.int64)
        self.cloudy_idx = tf.constant(cloudy_idx, dtype=tf.int64)
        self.rainy_idx = tf.constant(rainy_idx, dtype=tf.int64)

        self.critical_penalty = tf.constant(critical_penalty, dtype=self.dtype)
        self.non_critical_penalty = tf.constant(non_critical_penalty, dtype=self.dtype)

        # Internal state variables for accumulating results
        self.total_penalty = self.add_weight(name='total_penalty', initializer='zeros', dtype=self.dtype)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state for a single batch of data.

        Args:
            y_true: True labels (expected in one-hot or index format).
            y_pred: Predicted labels (usually the output of a softmax layer).
            sample_weight: Optional weights for samples.
        """
        # 1. Get predicted class indices
        y_pred_indices = tf.argmax(y_pred, axis=-1)

        # 2. Get true class indices
        # Check if y_true is one-hot or indices
        if tf.rank(y_true) == tf.rank(y_pred): # Likely one-hot
            y_true_indices = tf.argmax(y_true, axis=-1)
        else: # Likely indices
            y_true_indices = tf.cast(y_true, dtype=tf.int64)
            # If y_true has an extra dimension (e.g., [batch, 1]), remove it
            if tf.rank(y_true_indices) > 1 and tf.shape(y_true_indices)[-1] == 1:
                 y_true_indices = tf.squeeze(y_true_indices, axis=-1)

        # Ensure indices have the same data type
        y_pred_indices = tf.cast(y_pred_indices, dtype=y_true_indices.dtype)

        # 3. Identify misclassifications
        is_misclassified = tf.not_equal(y_true_indices, y_pred_indices)

        # 4. Identify samples belonging to critical classes (clear=0 or rainy=2)
        is_critical_class = tf.logical_or(
            tf.equal(y_true_indices, self.clear_idx),
            tf.equal(y_true_indices, self.rainy_idx)
        )
        # Identify samples belonging to the non-critical class (cloudy=1)
        is_non_critical_class = tf.equal(y_true_indices, self.cloudy_idx)

        # 5. Calculate penalty for each sample in the batch
        # Penalty = critical_penalty if (misclassified AND critical class)
        penalty_critical = tf.cast(tf.logical_and(is_misclassified, is_critical_class), self.dtype) * self.critical_penalty
        # Penalty = non_critical_penalty if (misclassified AND non-critical class)
        penalty_non_critical = tf.cast(tf.logical_and(is_misclassified, is_non_critical_class), self.dtype) * self.non_critical_penalty

        # Sum of penalties for the batch (penalty=0 for correctly classified samples)
        batch_penalty = penalty_critical + penalty_non_critical

        # Account for sample weights if available
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            # Adjust weight shape if necessary
            if tf.rank(sample_weight) > tf.rank(batch_penalty):
                 sample_weight = tf.squeeze(sample_weight, axis=-1)
            batch_penalty *= sample_weight

        # 6. Update total penalty and sample count
        self.total_penalty.assign_add(tf.reduce_sum(batch_penalty))
        num_samples_in_batch = tf.cast(tf.size(y_true_indices), self.dtype)
        self.count.assign_add(num_samples_in_batch)

    def result(self):
        """Calculates and returns the final metric value (average penalty)."""
        return tf.math.divide_no_nan(self.total_penalty, self.count)

    def reset_state(self):
        """Resets the internal states of the metric (called at the start of each epoch)."""
        self.total_penalty.assign(0.0)
        self.count.assign(0.0)

# Optional: Create an instance of the metric immediately with default parameters from config
# import config
# default_weather_penalty_metric = WeightedWeatherPenalty(
#     clear_idx=config.CLEAR_IDX,
#     cloudy_idx=config.CLOUDY_IDX,
#     rainy_idx=config.RAINY_IDX,
#     critical_penalty=config.CRITICAL_PENALTY,
#     non_critical_penalty=config.NON_CRITICAL_PENALTY
# )