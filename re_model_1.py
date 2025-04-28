# re_model_1.py
# Previous filename was train.py
"""Main script for fine-tuning a previously saved Keras model."""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model # Import Model as well
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D # Import necessary layers
from keras.regularizers import l2
from keras import mixed_precision # Commented out
# from keras.applications import EfficientNetB0, MobileNetV2, DenseNet121, ResNet50 # Not used directly for building base here
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
    """Loads a pre-trained model, modifies its classifier, and prepares for fine-tuning."""
    # Note: This function's logic for modifying the model is specific and might need adjustment
    # depending on the exact structure of 'best_model.keras'. It assumes a Sequential model.
    model = None # Initialize model to None
    try:
        print("Attempting to load model 'best_model.keras'...")
        # Ensure the custom metric definition is available
        custom_objects = {'WeightedWeatherPenalty': metrics.WeightedWeatherPenalty}
        # Load your saved model
        # model = keras.models.load_model('new_best_model.keras', custom_objects=custom_objects)
        model = keras.models.load_model('best_model.keras', custom_objects=custom_objects)
        print("Model loaded successfully.")

        # --- Modify the loaded Sequential model ---
        # Rename layers to avoid potential conflicts if model is reused
        for layer in model.layers:
            layer._name = f"loaded_{layer.name}"

        # Create a new Sequential model excluding the last layer of the loaded model
        # This effectively removes the original classifier head
        model = Sequential(model.layers[:-1])
        print("Removed the last layer of the loaded model.")

    except Exception as e: # Catch specific exceptions if possible (e.g., FileNotFoundError, ValueError)
        print(f"ERROR loading model: {e}")
        print("Could not load 'best_model.keras'. Please ensure the file exists and custom objects are correct.")
        # Depending on requirements, you might want to exit or raise the error
        # raise e
        return None # Return None to indicate failure

    # --- Freezing Strategy for Fine-tuning ---
    # Freeze layers of the loaded base part (all except the new ones to be added)
    print("Freezing layers of the loaded base model part...")
    # Freeze layers up to a certain depth (e.g., all but the last 10 of the loaded part)
    num_layers_to_fine_tune_base = 10 # Example: Fine-tune last 10 layers *of the loaded part*
    for layer in model.layers[:-num_layers_to_fine_tune_base]:
        layer.trainable = False
    for layer in model.layers[-num_layers_to_fine_tune_base:]:
        layer.trainable = True
    print(f"Base layers frozen except for the last {num_layers_to_fine_tune_base}.")

    # --- Add new classifier layers ---
    print("Adding new classifier layers...")
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='new_dense'))
    model.add(Dropout(0.3, name='new_dropout'))
    model.add(Dense(3, activation='softmax', name='new_output')) # Assuming 3 classes
    print("New layers added.")

    if model:
        model.summary() # Display the final architecture
    return model

def main():
    """Main function to run the fine-tuning and evaluation process."""
    print("Starting fine-tuning script...")
    setup_environment()

    # 1. Data Preparation
    print("\n--- Data Preparation ---")
    train_generator = data_utils.create_train_generator()
    validation_generator = data_utils.create_validation_generator()
    num_classes = train_generator.num_classes # Get number of classes from generator

    # Optional: Ensure class indices in config match the generator
    # Validation can be added for config.CLEAR_IDX etc. vs train_generator.class_indices

    # 2. Model Building / Loading / Modification
    print("\n--- Model Building / Loading ---")
    model = build_model() # build_model assumes 3 classes in the new head

    if model is None:
        print("Model building/loading failed. Exiting.")
        sys.exit(1) # Exit script if model loading failed

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

    # Use a lower learning rate for fine-tuning (optional but recommended)
    # fine_tune_lr = config.LEARNING_RATE / 10 # Example: 1/10th of original LR
    # print(f"Compiling model for fine-tuning with LR: {fine_tune_lr}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), # Using original LR here, adjust if needed
        loss='categorical_crossentropy',
        metrics=model_metrics
    )
    print("Model compiled.")

    # 4. Callback Preparation
    print("\n--- Callback Preparation ---")
    # Monitor the custom validation metric
    monitor_metric = f'val_{weather_penalty.name}'
    # Adjust checkpoint path if saving the fine-tuned model
    training_callbacks = cb.get_callbacks(monitor_metric=monitor_metric, model_checkpoint=False, checkpoint_path='best_finetuned_model.keras')
    print(f"Callbacks prepared. Monitored metric: {monitor_metric}")

    # 5. Model Fine-tuning
    print("\n--- Starting Fine-tuning ---")
    # Use the number of epochs specified in the config file
    print(f"Fine-tuning for {config.EPOCHS} epochs (from config)...")

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config.EPOCHS, # Use value from config (REVERTED)
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=config.CLASS_WEIGHTS, # Use class weights from config
        callbacks=training_callbacks,
        verbose=1
    )
    print("--- Fine-tuning Finished ---")

    # 6. Model Evaluation and Visualization
    print("\n--- Evaluation and Visualization ---")
    # Visualize training history
    evaluation_utils.plot_training_history(history, metrics_to_plot=['accuracy', 'loss', weather_penalty.name, 'precision', 'recall', 'auc'], name='re_model_1')
    # Perform full evaluation on the validation set
    evaluation_utils.evaluate_model(model, validation_generator, custom_metric_name=weather_penalty.name, name='re_model_1')

    # Optional: Save the fine-tuned model
    # final_model_path = "final_finetuned_model.keras"
    # model.save(final_model_path)
    # print(f"Final fine-tuned model saved to: {final_model_path}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()