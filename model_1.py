# model_1.py (Refactored Simple CNN)
# Previous filename was train.py
"""Main script for training model #1 (Simple CNN)."""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras.regularizers import l2
from keras import mixed_precision
import os
import sys

# Add the directory containing modules to PYTHONPATH if necessary
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

# Import custom modules
import config
import data_utils
import metrics
import callbacks as cb # Use an alias to avoid collision with keras.callbacks
import evaluation_utils

def setup_environment():
    """Configures the TensorFlow environment (GPU, Mixed Precision)."""
    # Mixed Precision Setup
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed Precision (mixed_float16) configured.")
    except Exception as e:
        print(f"Failed to configure Mixed Precision: {e}")

    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable dynamic memory growth for the first GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"GPU found: {gpus[0]}. Enabled dynamic memory growth.")
            # Optional: Set memory limit
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)] # e.g., 4GB
            # )
            # print("GPU memory limit set (if uncommented).")
        except RuntimeError as e:
            print(f"Error during GPU configuration: {e}")
    else:
        print("No GPU found. Training will run on CPU.")

def build_model_cnn(input_shape, num_classes):
    """Builds a simple CNN model (similar structure to the original script)."""
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
        Dense(num_classes, activation='softmax') # Output layer using num_classes
    ], name="Simple_CNN")
    print("Simple CNN model built.")
    model.summary()
    return model

def main():
    """Main function to run the training and evaluation process."""
    print("Starting training script...")
    setup_environment()

    # 1. Data Preparation
    print("\n--- Data Preparation ---")
    train_generator = data_utils.create_train_generator()
    validation_generator = data_utils.create_validation_generator()
    num_classes = train_generator.num_classes # Get number of classes from the generator

    # Optional: Ensure class indices in config match the generator
    # Validation can be added for config.CLEAR_IDX etc. vs train_generator.class_indices

    # 2. Model Building
    print("\n--- Model Building ---")
    model = build_model_cnn(input_shape=config.IMG_SHAPE, num_classes=num_classes)

    # 3. Metrics Preparation and Compilation
    print("\n--- Model Compilation ---")
    # Create instance of the custom metric
    weather_penalty = metrics.WeightedWeatherPenalty(
        clear_idx=config.CLEAR_IDX, # Use value from config
        cloudy_idx=config.CLOUDY_IDX, # Use value from config
        rainy_idx=config.RAINY_IDX, # Use value from config
        critical_penalty=config.CRITICAL_PENALTY, # Use value from config
        non_critical_penalty=config.NON_CRITICAL_PENALTY # Use value from config
    )

    # Define the list of metrics to use during training and evaluation
    model_metrics = [
        'accuracy',
        weather_penalty, # Add the custom metric instance
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=model_metrics
    )
    print("Model compiled.")

    # 4. Callback Preparation
    print("\n--- Callback Preparation ---")
    # Monitor the custom metric for early stopping and LR reduction
    monitor_metric = f'val_{weather_penalty.name}'
    # Get callbacks, disable ModelCheckpoint if not needed for this run
    training_callbacks = cb.get_callbacks(monitor_metric=monitor_metric, model_checkpoint=True, checkpoint_path='new_best_model.keras')
    print(f"Callbacks prepared. Monitored metric: {monitor_metric}")

    # 5. Model Training
    print("\n--- Starting Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config.EPOCHS, # Use number of epochs from config
        validation_data=validation_generator,
        validation_steps=len(validation_generator), # Specify validation steps for clarity
        class_weight=config.CLASS_WEIGHTS, # Use class weights from config
        callbacks=training_callbacks,
        verbose=1
    )
    print("--- Training Finished ---")

    # 6. Model Evaluation and Visualization
    print("\n--- Evaluation and Visualization ---")
    # Visualize training history
    evaluation_utils.plot_training_history(history, metrics_to_plot=['accuracy', 'loss', weather_penalty.name, 'precision', 'recall', 'auc'], name='model_1')

    # Perform full evaluation on the validation set
    evaluation_utils.evaluate_model(model, validation_generator, custom_metric_name=weather_penalty.name, name='model_1')

    # Optional: Save the final model
    # final_model_path = "final_cnn_model.keras"
    # model.save(final_model_path)
    # print(f"Final model saved to: {final_model_path}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()