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


setup_environment()

print("Attempting to load model 'best_model.keras'...")
# Ensure the custom metric definition is available
custom_objects = {'WeightedWeatherPenalty': metrics.WeightedWeatherPenalty}
# Load your saved model
# model = keras.models.load_model('new_best_model.keras', custom_objects=custom_objects)
model = keras.models.load_model('best_model.keras', custom_objects=custom_objects)
print("Model loaded successfully.")


weather_penalty = metrics.WeightedWeatherPenalty(
        clear_idx=config.CLEAR_IDX, # Use value from config
        cloudy_idx=config.CLOUDY_IDX, # Use value from config
        rainy_idx=config.RAINY_IDX, # Use value from config
        critical_penalty=config.CRITICAL_PENALTY, # Use value from config
        non_critical_penalty=config.NON_CRITICAL_PENALTY # Use value from config
    )

test_generator = data_utils.create_test_generator()

evaluation_utils.evaluate_model(model, test_generator, custom_metric_name=weather_penalty.name, name='test_model_1')