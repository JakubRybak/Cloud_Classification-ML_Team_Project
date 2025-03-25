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
import shutil



# ---------------------------------------------------------------------------------------------------
raw_train_dir = "raw_cloud_dataset/train"
raw_test_dir = "raw_cloud_dataset/test"
raw_val_dir = "raw_cloud_dataset/val"

processed_train_dir = "cloud_dataset/train"
processed_val_dir = "cloud_dataset/val"

img_width, img_heigth = 150, 150
batch_size = 32

if os.path.exists("cloud_dataset"):
    shutil.rmtree("cloud_dataset")
os.makedirs("cloud_dataset", exist_ok=True)

shutil.copytree("raw_cloud_dataset/train", "cloud_dataset/train")
shutil.copytree("raw_cloud_dataset/val", "cloud_dataset/val")
# ---------------------------------------------------------------------------------------------------



# CHECKING IF ALL IMAGES ARE OK
# ---------------------------------------------------------------------------------------------------
def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        print(f"Uszkodzony obraz {file_path}, błąd {e}")
        img.close()
        os.remove(file_path)
        return False
def check_rgb(file_path):
    img = Image.open(file_path)
    if img.mode != "RGB":
        print(file_path)
        img.close()
        os.remove(file_path)
        return False
    return True

count_bad = 0
count_not_rgb = 0
for root, dir, files in os.walk(processed_train_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not check_image(file_path):
            count_bad += 1
        if not check_rgb(file_path):
            count_not_rgb += 1
            
print("There are this many bad images: ", count_bad)
print("There are this many images not in RGB: ", count_not_rgb)
# os.remove("cloud_dataset/train/rainy/033.jpg")
# Removing ai image
os.remove("cloud_dataset/train/Rainy/original_size_img_77_jpg.rf.c3154c897671512abd8b90ebdb1b377c.jpg")
print()
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

size_list = check_image_sizes(processed_train_dir)

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






# DEALING WITH OUTLIERS
# ---------------------------------------------------------------------------------------------------
def is_too_small(image):
    width, height = image.size[0], image.size[1]
    return width <= 100 or height <= 100 
def is_too_big(image):
    width, height = image.size[0], image.size[1]
    return width >= 700 or height >= 550
# ---------------------------------------------------------------------------------------------------

# PREPROCESSING FUNCTIONS
# ---------------------------------------------------------------------------------------------------
def preprocessing_train(processed_train_dir):
    for root, _, files in os.walk(processed_train_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if_check_image = check_image(file_path)
            if not if_check_image:
                continue
            if_check_rgb = check_rgb(file_path)
            if not if_check_rgb:
                continue
            with Image.open(file_path) as img:
                if_to_small = is_too_small(img)
            if if_to_small:
                os.remove(file_path)
                continue
            with Image.open(file_path) as img:
                if is_too_big(img):
                    ratio = min(700/img.width, 550/img.height)
                    new_size = (int(img.width*ratio), int(img.height*ratio))
                    img.resize(new_size, Image.LANCZOS).save(file_path)

preprocessing_train(processed_train_dir)

def preprocessing_val(processed_val_dir):
    for root, _, files in os.walk(processed_val_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if_check_image = check_image(file_path)
            if not if_check_image:
                continue
            if_check_rgb = check_rgb(file_path)
            if not if_check_rgb:
                continue
            with Image.open(file_path) as img:
                if is_too_big(img):
                    ratio = min(700/img.width, 550/img.height)
                    new_size = (int(img.width*ratio), int(img.height*ratio))
                    img.resize(new_size, Image.LANCZOS).save(file_path)
preprocessing_val(processed_val_dir)
# ---------------------------------------------------------------------------------------------------




# GATHERING DATA INFORMATION
# ---------------------------------------------------------------------------------------------------
train_data = []

x = 0
classes = os.listdir(processed_train_dir)
for class_name in classes:
    class_dir = os.path.join(raw_train_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class {class_name}: {num_images} images")
    train_data.append([class_name, num_images, "Train"])
    x += num_images
train_data.append(["All", x, "Train"])
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
