import tensorflow as tf
import os
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard  # Removed ModelCheckpoint
from keras.regularizers import l2
from keras_tuner import RandomSearch, HyperModel
from keras.applications import EfficientNetB0, MobileNetV2
from keras.models import Model
from keras import mixed_precision
import datetime

# Define directories for raw and processed datasets
raw_train_dir = "raw_cloud_dataset/train"
raw_test_dir = "raw_cloud_dataset/test"
raw_val_dir = "raw_cloud_dataset/val"

processed_train_dir = "cloud_dataset/train"
processed_val_dir = "cloud_dataset/val"

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 16

# Configure mixed precision training to leverage Tensor Cores on compatible GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# GPU Configuration - Dynamic memory allocation and optional memory limit
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 1. Enable dynamic memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # 2. Optional: Set a memory limit (e.g., 90% of available memory).
        #    Note: Effectiveness may vary.
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]  # Example: 4GB limit
        # )
    except RuntimeError as e:
        print(e)

# IMAGE DATA GENERATORS
# ---------------------------------------------------------------------------------------------------
# Create a data generator for training data with augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to be between 0 and 1
    rotation_range=20,  # Randomly rotate images up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images horizontally
    brightness_range=[0.7, 1.3],  # Randomly adjust brightness
    fill_mode="nearest"  # Fill in newly created pixels with the nearest value
)

# Create a generator for the training data, reading from the specified directory
train_generator = train_datagen.flow_from_directory(
    processed_train_dir,
    target_size=(img_width, img_height),  # Resize images to the target size
    batch_size=batch_size,
    class_mode="categorical"  # Use categorical labels for multi-class classification
)


# Create a data generator for validation data (only rescaling, no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1./255  # Normalize pixel values
)

# Create a generator for the validation data
validation_generator = validation_datagen.flow_from_directory(
    processed_val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical"
)
print()
# ---------------------------------------------------------------------------------------------------

# Define a directory for TensorBoard logs, including a timestamp
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Define callbacks for training
callbacks = [
    EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=15,  # Stop training if no improvement after 15 epochs
        restore_best_weights=True  # Restore the weights of the best epoch
    ),
    # ModelCheckpoint(
    #     'best_model.h5',  # Save the best model to this file
    #     monitor='val_accuracy',  # Monitor validation accuracy
    #     save_best_only=True  # Only save the best model
    # ),
    ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.2,  # Reduce learning rate by this factor
        patience=5,  # Reduce LR if no improvement after 5 epochs
        min_lr=1e-6  # Minimum learning rate
    ),
    TensorBoard(
        log_dir=log_dir,  # Log directory
        histogram_freq=1,  # Compute histogram frequencies for visualization
        write_graph=True,  # Write the computation graph to TensorBoard
        write_images=False,  # Do not write images to TensorBoard
        update_freq='epoch',  # Log metrics every epoch
        profile_batch=0  # Profile the first batch (0 means no profiling)
    )
]


# HYPERPARAMETER TUNING WITH KERAS TUNER
# ---------------------------------------------------------------------------------------------------
def model_builder(hp):
    """
    Builds the CNN model with tunable hyperparameters.

    Args:
        hp: HyperParameters object from keras_tuner.

    Returns:
        A compiled Keras model.
    """
    model = Sequential()
    # Input Layer
    model.add(Conv2D(hp.Choice('input_filters', values=[32, 64]), (3, 3), padding='same', input_shape=(img_width, img_height, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(hp.Choice('input_filters', values=[32, 64]), (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(hp.Float('input_dropout', min_value=0.1, max_value=0.3, step=0.1)))

    # Hidden Layers
    for i in range(hp.Int('num_conv_layers', min_value=1, max_value=3)):
        model.add(Conv2D(hp.Choice(f'conv_{i}_filters', values=[64, 128, 256]), (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(hp.Choice(f'conv_{i}_filters', values=[64, 128, 256]), (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(hp.Float(f'conv_{i}_dropout', min_value=0.1, max_value=0.5, step=0.1)))

    # Classifier
    model.add(Flatten())
    model.add(Dense(hp.Choice('dense_units', values=[128, 256, 512]), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dense_dropout', min_value=0.3, max_value=0.6, step=0.1)))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    return model


# Instantiate the tuner
tuner = RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=20,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of times to train each model
    directory='tuner_dir',  # Directory to store tuning logs
    project_name='cloud_classification'
)

# Perform the hyperparameter search
tuner.search(
    train_generator,
    validation_data=validation_generator,
    callbacks=callbacks  # Callbacks now exclude ModelCheckpoint
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Print a summary of the best model
best_model.summary()

# ---------------------------------------------------------------------------------------------------


# 5. Train the best model
history = best_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Automatically calculate steps per epoch
    epochs=100,  # Maximum number of epochs (EarlyStopping will stop if necessary)
    validation_data=validation_generator,  # Use validation data for monitoring
    callbacks=callbacks,  # Callbacks now exclude ModelCheckpoint
    verbose=1  # Display training progress
)


# Plot training history
plt.figure(figsize=(15, 5))

# Plot Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

# Plot Precision
plt.subplot(1, 3, 2)
plt.plot(history.history['precision'], label='Train')
plt.plot(history.history['val_precision'], label='Validation')
plt.title('Precision')
plt.legend()

# Plot Recall
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Train')
plt.plot(history.history['val_recall'], label='Validation')
plt.title('Recall')
plt.legend()

plt.tight_layout()
plt.show()

# Print final validation report
print("\nBest Validation Results:")
print(f"• Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"• Precision: {max(history.history['val_precision']):.4f}")
print(f"• Recall: {max(history.history['val_recall']):.4f}")

# Optional: Instructions for running TensorBoard
# print(f"\nTo run TensorBoard, execute in the terminal:")
# print(f"tensorboard --logdir {log_dir}")
# print("Then open in your browser: http://localhost:6006")
# ---------------------------------------------------------------------------------------------------