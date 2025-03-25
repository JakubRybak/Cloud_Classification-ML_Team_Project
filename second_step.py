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


raw_train_dir = "raw_cloud_dataset/train"
raw_test_dir = "raw_cloud_dataset/test"
raw_val_dir = "raw_cloud_dataset/val"

processed_train_dir = "cloud_dataset/train"
processed_val_dir = "cloud_dataset/val"

img_width, img_heigth = 150, 150
batch_size = 32




#GENERATORS
# ---------------------------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
)

train_generator = train_datagen.flow_from_directory(
    processed_train_dir,
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)


validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    processed_val_dir,
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)
print()
# ---------------------------------------------------------------------------------------------------




#CHECKING IF EVERYTHING WORKS
# ---------------------------------------------------------------------------------------------------
images, labels = next(train_generator)

plt.figure(figsize=(10, 10))
for i in range(9):  # Wyświetl 9 obrazów
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Klasa: {labels[i]}")
    plt.axis("off")

print(f"Min: {images.min()}, Max: {images.max()}")

images2, labels2 = next(validation_generator)
print(f"Min: {images2.min()}, Max: {images2.max()}")

plt.close()
# ---------------------------------------------------------------------------------------------------




