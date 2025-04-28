# data_utils.py
"""Helper functions for loading and processing image data using Keras ImageDataGenerator."""

from keras.preprocessing.image import ImageDataGenerator
import config # Import project configuration settings

def create_train_generator(directory=config.PROCESSED_TRAIN_DIR,
                           target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
                           batch_size=config.BATCH_SIZE,
                           class_mode="categorical"):
    """
    Creates a training data generator with real-time data augmentation.

    Args:
        directory (str): Path to the training data directory.
        target_size (tuple): Dimensions to resize images to (height, width).
        batch_size (int): Size of the batches of data.
        class_mode (str): Mode for yielding labels ('categorical', 'binary', etc.).

    Returns:
        keras.preprocessing.image.DirectoryIterator: A generator yielding tuples of (x, y).
    """
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,         # Rescale pixel values from [0, 255] to [0, 1].
    #     rotation_range=5,      # Randomly rotate images by up to 5 degrees.
    #     width_shift_range=0.05,  # Randomly shift images horizontally.
    #     height_shift_range=0.05, # Randomly shift images vertically.
    #     shear_range=0.1,        # Apply shear transformations.
    #     zoom_range=0.1,         # Randomly zoom into images.
    #     horizontal_flip=True,   # Randomly flip images horizontally.
    #     brightness_range=[0.95, 1.05], # Randomly adjust brightness.
    #     fill_mode="nearest"     # Strategy for filling newly created pixels.
    # )
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Rescale pixel values from [0, 255] to [0, 1].
        rotation_range=20,      # Randomly rotate images by up to 20 degrees.
        width_shift_range=0.2,  # Randomly shift images horizontally.
        height_shift_range=0.2, # Randomly shift images vertically.
        shear_range=0.2,        # Apply shear transformations.
        zoom_range=0.2,         # Randomly zoom into images.
        horizontal_flip=True,   # Randomly flip images horizontally.
        brightness_range=[0.7, 1.3], # Randomly adjust brightness.
        fill_mode="nearest"     # Strategy for filling newly created pixels.
    )

    print(f"Creating training generator from directory: {directory}")
    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=target_size, # Resize images to the specified dimensions.
        batch_size=batch_size,
        class_mode=class_mode    # Type of labels to return.
        # shuffle=True is the default and generally desired for training.
    )

    print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes.")
    print("Class indices:", train_generator.class_indices)
    return train_generator

def create_validation_generator(directory=config.PROCESSED_VAL_DIR,
                                target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
                                batch_size=config.BATCH_SIZE,
                                class_mode="categorical"):
    """
    Creates a validation data generator (only rescaling, no augmentation).

    Args:
        directory (str): Path to the validation data directory.
        target_size (tuple): Dimensions to resize images to (height, width).
        batch_size (int): Size of the batches of data.
        class_mode (str): Mode for yielding labels ('categorical', 'binary', etc.).

    Returns:
        keras.preprocessing.image.DirectoryIterator: A generator yielding tuples of (x, y).
    """
    # Only rescale validation data, do not apply augmentation.
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Creating validation generator from directory: {directory}")
    validation_generator = validation_datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False # Important for consistent evaluation and confusion matrix calculation!
    )

    print(f"Found {validation_generator.samples} images belonging to {validation_generator.num_classes} classes.")
    print("Class indices:", validation_generator.class_indices)

    # Optional: Check for class index consistency between train and validation generators
    # if 'train_generator' in locals() and train_generator.class_indices != validation_generator.class_indices:
    #    print("WARNING: Class indices of training and validation generators differ!")

    return validation_generator

def create_test_generator(directory=config.PROCESSED_TEST_DIR,
                                target_size=(config.IMG_WIDTH, config.IMG_HEIGHT),
                                batch_size=config.BATCH_SIZE,
                                class_mode="categorical"):
    """
    Creates a validation data generator (only rescaling, no augmentation).

    Args:
        directory (str): Path to the validation data directory.
        target_size (tuple): Dimensions to resize images to (height, width).
        batch_size (int): Size of the batches of data.
        class_mode (str): Mode for yielding labels ('categorical', 'binary', etc.).

    Returns:
        keras.preprocessing.image.DirectoryIterator: A generator yielding tuples of (x, y).
    """
    # Only rescale validation data, do not apply augmentation.
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Creating validation generator from directory: {directory}")
    validation_generator = validation_datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False # Important for consistent evaluation and confusion matrix calculation!
    )

    print(f"Found {validation_generator.samples} images belonging to {validation_generator.num_classes} classes.")
    print("Class indices:", validation_generator.class_indices)

    # Optional: Check for class index consistency between train and validation generators
    # if 'train_generator' in locals() and train_generator.class_indices != validation_generator.class_indices:
    #    print("WARNING: Class indices of training and validation generators differ!")

    return validation_generator
# TODO: Consider adding create_test_generator if a separate test set is used.
# def create_test_generator(...):
#     ...