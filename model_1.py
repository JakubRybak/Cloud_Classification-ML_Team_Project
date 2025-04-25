# train.py
"""Główny skrypt do trenowania modelu nr 1 (CNN)."""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras.regularizers import l2
from keras import mixed_precision
import os
import sys

# Dodaj ścieżkę do katalogu z modułami, jeśli nie jest w PYTHONPATH
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

# Importuj własne moduły
import config
import data_utils
import metrics
import callbacks as cb # Używamy aliasu, żeby nie kolidowało z keras.callbacks
import evaluation_utils

def setup_environment():
    """Konfiguruje środowisko TensorFlow (GPU, Mixed Precision)."""
    # Mixed Precision
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed Precision (mixed_float16) skonfigurowane.")
    except Exception as e:
        print(f"Nie udało się skonfigurować Mixed Precision: {e}")

    # Konfiguracja GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Znaleziono GPU: {gpus[0]}. Włączono dynamiczny wzrost pamięci.")
            # Opcjonalne: Ustawienie limitu pamięci
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)] # np. 4GB
            # )
            # print("Ustawiono limit pamięci GPU (jeśli odkomentowane).")
        except RuntimeError as e:
            print(f"Błąd podczas konfiguracji GPU: {e}")
    else:
        print("Nie znaleziono GPU. Trening będzie odbywał się na CPU.")

def build_model_cnn(input_shape, num_classes):
    """Buduje prosty model CNN (jak w oryginalnym pliku)."""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(), Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(), Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(), Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.6),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax') # Użyj num_classes
    ], name="Simple_CNN") #
    print("Model CNN zbudowany.")
    model.summary()
    return model

def main():
    """Główna funkcja uruchamiająca proces."""
    print("Rozpoczynanie skryptu treningowego...")
    setup_environment()

    # 1. Przygotowanie danych
    print("\n--- Przygotowanie Danych ---")
    train_generator = data_utils.create_train_generator()
    validation_generator = data_utils.create_validation_generator()
    num_classes = train_generator.num_classes # Pobierz liczbę klas z generatora

    # Upewnij się, że indeksy klas w config pasują do generatora
    # Można dodać walidację config.CLEAR_IDX itp. vs train_generator.class_indices

    # 2. Budowa modelu
    print("\n--- Budowa Modelu ---")
    model = build_model_cnn(input_shape=config.IMG_SHAPE, num_classes=num_classes)

    # 3. Przygotowanie metryk i kompilacja
    print("\n--- Kompilacja Modelu ---")
    # Utwórz instancję niestandardowej metryki
    weather_penalty = metrics.WeightedWeatherPenalty(
        clear_idx=config.CLEAR_IDX, # Użyj wartości z config
        cloudy_idx=config.CLOUDY_IDX,
        rainy_idx=config.RAINY_IDX,
        critical_penalty=config.CRITICAL_PENALTY,
        non_critical_penalty=config.NON_CRITICAL_PENALTY
    )

    # Zdefiniuj listę metryk do użycia
    model_metrics = [
        'accuracy',
        weather_penalty, # Dodaj instancję metryki
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=model_metrics
    ) #
    print("Model skompilowany.")

    # 4. Przygotowanie callbacków
    print("\n--- Przygotowanie Callbacków ---")
    # Monitoruj niestandardową metrykę
    monitor_metric = f'val_{weather_penalty.name}'
    training_callbacks = cb.get_callbacks(monitor_metric=monitor_metric, model_checkpoint=False) # Wyłącz ModelCheckpoint, jeśli niepotrzebny
    print(f"Callbacki przygotowane. Monitorowana metryka: {monitor_metric}")

    # 5. Trening modelu
    print("\n--- Rozpoczęcie Treningu ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config.EPOCHS, # Użyj wartości z config
        validation_data=validation_generator,
        validation_steps=len(validation_generator), # Można dodać dla pewności
        class_weight=config.CLASS_WEIGHTS, # Użyj wartości z config
        callbacks=training_callbacks,
        verbose=1
    ) #
    print("--- Trening Zakończony ---")

    # 6. Ewaluacja modelu i wizualizacja
    print("\n--- Ewaluacja i Wizualizacja ---")
    # Wizualizacja historii treningu
    evaluation_utils.plot_training_history(history, metrics_to_plot=['accuracy', 'loss', weather_penalty.name, 'precision', 'recall'])

    # Pełna ewaluacja na zbiorze walidacyjnym
    evaluation_utils.evaluate_model(model, validation_generator, custom_metric_name=weather_penalty.name)

    # Opcjonalnie: Zapisz ostateczny model
    # final_model_path = "final_cnn_model.keras"
    # model.save(final_model_path)
    # print(f"Ostateczny model zapisany w: {final_model_path}")

    print("\nSkrypt zakończył działanie.")

if __name__ == "__main__":
    main()