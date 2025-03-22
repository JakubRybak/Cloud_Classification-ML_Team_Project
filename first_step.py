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




# GATHERING DATA INFORMATION
# ---------------------------------------------------------------------------------------------------
train_data = []

print("Train dataset:")
print("ALL: ", train_generator.samples)
train_data.append(["All", train_generator.samples, "Train"])

classes = os.listdir(train_dir)
for class_name in classes:
    class_dir = os.path.join(train_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class {class_name}: {num_images} images")
    train_data.append([class_name, num_images, "Train"])
print()
# ---------------------------------------------------------------------------------------------------




# DATA PRESENTATION
# ---------------------------------------------------------------------------------------------------
train_data_to_graph = pd.DataFrame(train_data, columns=["Class", "Count", "Dataset"])


train_data_to_graph.plot(kind='bar', x="Class", y="Count", figsize=(10,7))
plt.title('Number of images in each class')
plt.xlabel('Class')
plt.ylabel('Number of images')
plt.close()
# ---------------------------------------------------------------------------------------------------


# CHECKING OUTLIERS
# ---------------------------------------------------------------------------------------------------
def check_image_sizes(directory):
    size_list = np.empty((0, 2))
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    size_list = np.append(size_list, [[img.size[0], img.size[1]]], axis = 0)
    return size_list

size_list = check_image_sizes('cloud_dataset/train')

# Basic stats   
# print(f"Size all {size_list.shape}")
# print(f"Mean {np.mean(size_list, axis=0)}")
# print(f"Minimum {np.min(size_list, axis=0)}")
# print(f"Maksimum {np.max(size_list, axis=0)}")


plt.figure(figsize=(12,6))

# Histogram for height
plt.subplot(2, 2, 1)  
plt.hist(size_list[:, 1], bins=35, color='blue', edgecolor='black')
plt.title('Height histogram')
plt.xlabel('Height')
plt.ylabel('Number of samples')


# Histogram for width
plt.subplot(2, 2, 2) 
plt.hist(size_list[:, 0], bins=35, color='green', edgecolor='black')
plt.title('Width histogram')
plt.xlabel('Width')
plt.ylabel('Number of samples')

width_condition_max = size_list[:, 0] < 700
height_condition_max = size_list[:, 1] < 550
conditions_max = width_condition_max & height_condition_max

width_condition_min = size_list[:, 0] > 100
height_condition_min = size_list[:, 1] > 100
conditions_min = width_condition_min & height_condition_min

conditions = conditions_max & conditions_min

not_outlayers = size_list[conditions]

print("All: ", size_list.shape[0])
print("Not outliers: ", not_outlayers.shape[0])
print()

plt.subplot(2, 2, 3)
plt.hist(not_outlayers[:, 1], bins=35, color="purple")
plt.title('Not outliers height histogram')
plt.xlabel('Height')
plt.ylabel('Number of samples')

plt.subplot(2, 2, 4)
plt.hist(not_outlayers[:, 0], bins=35, color="purple")
plt.title('Not outliers width histogram')
plt.xlabel('Width')
plt.ylabel('Number of samples')

plt.tight_layout()
plt.close()
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
plt.close()
# ---------------------------------------------------------------------------------------------------
