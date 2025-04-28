# config.py
"""Configuration for paths, image dimensions, and training parameters."""

import os

# --- Data Paths ---
RAW_DATA_DIR = "raw_cloud_dataset"
PROCESSED_DATA_DIR = "cloud_dataset"
RAW_TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
RAW_TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
RAW_VAL_DIR = os.path.join(RAW_DATA_DIR, "val")
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
PROCESSED_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
PROCESSED_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")


# --- Image Parameters ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3) # Expected image shape (height, width, channels)

# --- Training Parameters ---
BATCH_SIZE = 16 # Number of samples per gradient update.
EPOCHS = 70     # Maximum number of epochs to train for.

# --- Logging ---
LOG_DIR_BASE = "logs/fit" # Base directory for TensorBoard logs.

# --- Class Weights ---
# Used to balance the loss function for imbalanced datasets.
# Can also be calculated dynamically in data_utils.py or train.py.
CLASS_WEIGHTS = {
    0: 0.4, # Example weight for class 0
    1: 0.2, # Example weight for class 1
    2: 0.4  # Example weight for class 2
}

# --- WeightedWeatherPenalty Metric Parameters ---
# Ensure these indices match the generator's class_indices output.
CLEAR_IDX = 0           # Index for the 'clear' class (considered critical).
CLOUDY_IDX = 1          # Index for the 'cloudy' class (considered non-critical).
RAINY_IDX = 2           # Index for the 'rainy' class (considered critical).
CRITICAL_PENALTY = 1.0      # Penalty for misclassifying a critical class.
NON_CRITICAL_PENALTY = 0.3  # Penalty for misclassifying a non-critical class.

# --- Callback Settings ---
EARLY_STOPPING_PATIENCE = 25 # Patience for EarlyStopping callback.
REDUCE_LR_PATIENCE = 5       # Patience for ReduceLROnPlateau callback.
REDUCE_LR_FACTOR = 0.2       # Factor for reducing learning rate in ReduceLROnPlateau.
MIN_LR = 1e-6                # Minimum learning rate for ReduceLROnPlateau.

# --- Optimizer Parameters ---
LEARNING_RATE = 0.001 # Initial learning rate for the optimizer.

NUM_CLASSES = 3