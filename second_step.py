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


train_dir = "cloud_dataset/train"
test_dir = "cloud_dataset/test"
val_dir = "cloud_dataset/val"

img_width, img_heigth = 150, 150
batch_size = 32




# CHECKING IF ALL IMAGES ARE OK
# ---------------------------------------------------------------------------------------------------
def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        print(f"Uszkodzony obraz {file_path}, błąd {e}")
        return False
def check_rgb(file_path):
    img = Image.open(file_path)
    if img.mode != "RGB":
        # print(file_path)
        return False
    return True

count_bad = 0
count_not_rgb = 0
for root, dir, files in os.walk(train_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not check_image(file_path):
            count_bad += 1
        if not check_rgb(file_path):
            count_not_rgb += 1
            
print("There are this many bad images: ", count_bad)
print("One image was not in RGB and has been removed")
print("There are this many images not in RGB: ", count_not_rgb)
print()
# ---------------------------------------------------------------------------------------------------





# REMOVING OUTLIERS
# ---------------------------------------------------------------------------------------------------
def is_valid(image):
    width, height = image.size[0], image.size[1]
    return width > 100 and width < 700 and height > 100 and height < 550

valid_images = []
classes = []

for root, dir, files in os.walk(train_dir):
    for file in files:
        file_path = os.path.join(root, file)
        with Image.open(file_path) as img:
            if is_valid(img):
                valid_images.append(file_path)
                classes.append(os.path.basename(root))

df = pd.DataFrame({'filename': valid_images, 'class': classes})
# ---------------------------------------------------------------------------------------------------





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

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    x_col = "filename",
    y_col = "class",
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)


validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)
print()
# ---------------------------------------------------------------------------------------------------






