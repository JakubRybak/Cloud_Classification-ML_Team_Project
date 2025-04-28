import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras.regularizers import l2
from keras import mixed_precision
import os

from keras import backend as K
import gc

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules
import config
import data_utils
import metrics
import callbacks as cb
import evaluation_utils


def setup_environment():
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed Precision (mixed_float16) configured.")
    except Exception as e:
        print(f"Failed to configure Mixed Precision: {e}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"GPU found: {gpus[0]}. Enabled dynamic memory growth.")
        except RuntimeError as e:
            print(f"Error during GPU configuration: {e}")
    else:
        print("No GPU found. Training will run on CPU.")


def build_model(hp):
    tf.keras.backend.clear_session()  # <-- pamiętaj, na samym początku!

    model = Sequential()

    model.add(Conv2D(
        filters=hp.Choice('filters_1', values=[32, 64]),
        kernel_size=(3, 3),
        padding='same',
        input_shape=config.IMG_SHAPE
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(
        filters=hp.Choice('filters_2', values=[32, 64]),
        kernel_size=(3, 3),
        padding='same'
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(hp.Choice('dropout_1', values=[0.3, 0.4, 0.5])))

    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice('dense_units', values=[128, 256]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.01, 0.1]))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout_2', values=[0.5, 0.6])))

    model.add(Dense(config.NUM_CLASSES, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    )

    weather_penalty = metrics.WeightedWeatherPenalty(
        clear_idx=config.CLEAR_IDX,
        cloudy_idx=config.CLOUDY_IDX,
        rainy_idx=config.RAINY_IDX,
        critical_penalty=config.CRITICAL_PENALTY,
        non_critical_penalty=config.NON_CRITICAL_PENALTY
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            weather_penalty,
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    return model


def main():
    print("Starting tuning script...")
    setup_environment()

    print("\n--- Data Preparation ---")
    train_generator = data_utils.create_train_generator()
    validation_generator = data_utils.create_validation_generator()

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='cnn_tuning'
    )

    tuner.search(train_generator,
                 epochs=3,
                 validation_data=validation_generator)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    for param in best_hps.values.keys():
        print(f"{param}: {best_hps.get(param)}")

    print("\n--- Visualization of tuning results ---")
    tuner.results_summary()

    # --- Pobieramy wszystkie triale z tunera ---
    # --- Pobieramy wszystkie triale ---
    all_trials = list(tuner.oracle.trials.values())

    # --- Zbieramy metryki ---
    val_accuracies = []
    weather_penalties = []
    trial_names = []

    for i, trial in enumerate(all_trials):
        if trial.status == 'COMPLETED':
            val_acc = trial.metrics.get_last_value('val_accuracy')
            weather_penalty = trial.metrics.get_last_value(
                'val_weighted_weather_penalty')

            if val_acc is not None and weather_penalty is not None:
                val_accuracies.append(val_acc)
                weather_penalties.append(weather_penalty)
                trial_names.append(f'Trial {i}')

    # --- Rysowanie wykresu Validation Accuracy ---
    plt.figure(figsize=(12, 6))
    plt.bar(trial_names, val_accuracies, color='skyblue')
    plt.xlabel('Trial')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy for Each Trial')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Rysowanie wykresu Weather Penalty ---
    plt.figure(figsize=(12, 6))
    plt.bar(trial_names, weather_penalties, color='salmon')
    plt.xlabel('Trial')
    plt.ylabel('Weighted Weather Penalty')
    plt.title('Weighted Weather Penalty for Each Trial')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    K.clear_session()
    gc.collect()
    tf.config.experimental.reset_memory_stats('GPU:0')

    print("\n--- Training the best model on full epochs ---")
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        train_generator,
        batch_size=4,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        callbacks=cb.get_callbacks(
            monitor_metric='val_accuracy', model_checkpoint=False),
        verbose=1
    )

    final_model_path = "best_trained_model.keras"
    best_model.save(final_model_path)
    print(f"Best model saved to: {final_model_path}")

    print("\n--- Plotting full training history for the best model ---")
    evaluation_utils.plot_training_history(history, metrics_to_plot=[
                                           'accuracy', 'loss', 'weighted_weather_penalty', 'precision', 'recall', 'auc'])
    plt.savefig('Figure_tuning.png')
    plt.show()


if __name__ == "__main__":
    main()
