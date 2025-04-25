# model_3.py
# Previous filename was train.py
"""Main script for training model #3 (MobileNetV2 Transfer Learning)."""

import tensorflow as tf
from tensorflow import keras
# from keras.models import Sequential # Not used directly
from keras.layers import Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D # Import necessary layers
# from keras.regularizers import l2 # Not used directly
from keras import mixed_precision # Commented out
from keras.applications import MobileNetV2 # Import MobileNetV2
from keras.models import Model
# from keras.applications import EfficientNetB0 # Not used here
import os
import sys

# Add the directory containing modules to PYTHONPATH if necessary
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

# Import custom modules
import config
import data_utils
import metrics
import callbacks as cb # Use an alias
import evaluation_utils

def setup_environment():
    """Configures the TensorFlow environment (GPU)."""
    # Mixed precision is commented out in this version.
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

def build_model():
    """Builds the model using MobileNetV2 as a base for transfer learning."""
    # Create the base MobileNetV2 model, pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(512, activation='relu')(x)  # Dense layer
    predictions = Dense(3, activation='softmax')(x) # Output layer for 3 classes

    # Define the complete model
    model = Model(inputs=base_model.input, outputs=predictions, name="MobileNetV2_Transfer")

    # --- Layer Freezing Strategy ---
    # Freeze all but the last N layers for fine-tuning
    num_layers_to_fine_tune = 20 # Example: Fine-tune the top 20 layers
    for layer in base_model.layers[:-num_layers_to_fine_tune]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers_to_fine_tune:]:
        layer.trainable = True

    print(f"MobileNetV2 base model loaded. Top {num_layers_to_fine_tune} layers set to trainable.")
    model.summary()
    return model

def main():
    """Main function to run the training and evaluation process."""
    print("Starting training script for MobileNetV2 model...")
    setup_environment()

    # 1. Data Preparation
    print("\n--- Data Preparation ---")
    train_generator = data_utils.create_train_generator()
    validation_generator = data_utils.create_validation_generator()
    num_classes = train_generator.num_classes # Get number of classes from generator

    # Optional: Ensure class indices in config match the generator
    # Validation can be added for config.CLEAR_IDX etc. vs train_generator.class_indices

    # 2. Model Building
    print("\n--- Model Building ---")
    model = build_model() # Assumes 3 classes

    # 3. Metrics Preparation and Compilation
    print("\n--- Model Compilation ---")
    # Create instance of the custom metric
    weather_penalty = metrics.WeightedWeatherPenalty(
        clear_idx=config.CLEAR_IDX, cloudy_idx=config.CLOUDY_IDX, rainy_idx=config.RAINY_IDX,
        critical_penalty=config.CRITICAL_PENALTY, non_critical_penalty=config.NON_CRITICAL_PENALTY
    )
    # Define the list of metrics
    model_metrics = [
        'accuracy', weather_penalty, keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=model_metrics
    )
    print("Model compiled.")

    # 4. Callback Preparation
    print("\n--- Callback Preparation ---")
    # Monitor the custom validation metric
    monitor_metric = f'val_{weather_penalty.name}'
    training_callbacks = cb.get_callbacks(monitor_metric=monitor_metric, model_checkpoint=False)
    print(f"Callbacks prepared. Monitored metric: {monitor_metric}")

    # 5. Model Training
    print("\n--- Starting Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config.EPOCHS, # Use number of epochs from config
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=config.CLASS_WEIGHTS, # Use class weights from config
        callbacks=training_callbacks,
        verbose=1
    )
    print("--- Training Finished ---")

    # 6. Model Evaluation and Visualization
    print("\n--- Evaluation and Visualization ---")
    # Visualize training history
    evaluation_utils.plot_training_history(history, metrics_to_plot=['accuracy', 'loss', weather_penalty.name, 'precision', 'recall'])
    # Perform full evaluation on the validation set
    evaluation_utils.evaluate_model(model, validation_generator, custom_metric_name=weather_penalty.name)

    # Optional: Save the final model
    # final_model_path = "final_mobilenet_model.keras"
    # model.save(final_model_path)
    # print(f"Final model saved to: {final_model_path}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()