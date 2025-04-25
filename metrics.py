# metrics.py
"""Niestandardowe metryki Keras."""

import tensorflow as tf

class WeightedWeatherPenalty(tf.keras.metrics.Metric):
    """
    Niestandardowa metryka kary ważonej dla klasyfikacji pogody.
   
    Kara jest wyższa za błędną klasyfikację klas 'clear' (0) i 'rainy' (2),
    a niższa za błędną klasyfikację klasy 'cloudy' (1).
    Niższa wartość metryki oznacza lepszy wynik.
    """
    def __init__(self, clear_idx=0, cloudy_idx=1, rainy_idx=2,
                 critical_penalty=10.0, non_critical_penalty=1.0,
                 name='weighted_weather_penalty', dtype=tf.float32):
        """
        Inicjalizuje metrykę.

        Args:
            clear_idx (int): Indeks klasy 'clear' (krytyczna).
            cloudy_idx (int): Indeks klasy 'cloudy' (niekrytyczna).
            rainy_idx (int): Indeks klasy 'rainy' (krytyczna).
            critical_penalty (float): Kara za błędną klasyfikację klasy krytycznej.
            non_critical_penalty (float): Kara za błędną klasyfikację klasy niekrytycznej.
            name (str): Nazwa metryki.
            dtype: Typ danych dla obliczeń metryki.
        """
        super().__init__(name=name, dtype=dtype)
        self.clear_idx = tf.constant(clear_idx, dtype=tf.int64)
        self.cloudy_idx = tf.constant(cloudy_idx, dtype=tf.int64)
        self.rainy_idx = tf.constant(rainy_idx, dtype=tf.int64)
        self.critical_penalty = tf.constant(critical_penalty, dtype=self.dtype)
        self.non_critical_penalty = tf.constant(non_critical_penalty, dtype=self.dtype)
        self.total_penalty = self.add_weight(name='total_penalty', initializer='zeros', dtype=self.dtype)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=self.dtype)
        #

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Aktualizuje stan metryki dla pojedynczej paczki danych."""
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        if tf.rank(y_true) == tf.rank(y_pred):
            y_true_indices = tf.argmax(y_true, axis=-1)
        else:
            y_true_indices = tf.cast(y_true, dtype=tf.int64)
            if tf.rank(y_true_indices) > 1 and tf.shape(y_true_indices)[-1] == 1:
                 y_true_indices = tf.squeeze(y_true_indices, axis=-1)

        y_pred_indices = tf.cast(y_pred_indices, dtype=y_true_indices.dtype)
        is_misclassified = tf.not_equal(y_true_indices, y_pred_indices)
        is_critical_class = tf.logical_or(
            tf.equal(y_true_indices, self.clear_idx),
            tf.equal(y_true_indices, self.rainy_idx)
        )
        is_non_critical_class = tf.equal(y_true_indices, self.cloudy_idx)
        penalty_critical = tf.cast(tf.logical_and(is_misclassified, is_critical_class), self.dtype) * self.critical_penalty
        penalty_non_critical = tf.cast(tf.logical_and(is_misclassified, is_non_critical_class), self.dtype) * self.non_critical_penalty
        batch_penalty = penalty_critical + penalty_non_critical

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            if tf.rank(sample_weight) > tf.rank(batch_penalty):
                 sample_weight = tf.squeeze(sample_weight, axis=-1)
            batch_penalty *= sample_weight

        self.total_penalty.assign_add(tf.reduce_sum(batch_penalty))
        num_samples_in_batch = tf.cast(tf.size(y_true_indices), self.dtype)
        self.count.assign_add(num_samples_in_batch)
        #

    def result(self):
        """Oblicza i zwraca końcową wartość metryki."""
        return tf.math.divide_no_nan(self.total_penalty, self.count) #

    def reset_state(self):
        """Resetuje wewnętrzne stany metryki."""
        self.total_penalty.assign(0.0)
        self.count.assign(0.0)
        #

# Możesz od razu stworzyć instancję metryki z domyślnymi parametrami z config
# import config
# default_weather_penalty_metric = WeightedWeatherPenalty(
#     clear_idx=config.CLEAR_IDX,
#     cloudy_idx=config.CLOUDY_IDX,
#     rainy_idx=config.RAINY_IDX,
#     critical_penalty=config.CRITICAL_PENALTY,
#     non_critical_penalty=config.NON_CRITICAL_PENALTY
# )