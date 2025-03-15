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



#SPRAWDZENIE CZY WSZYSTKIE OBRAZY SA GIT--------------------------------------------------------
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
        # img = img.convert("RGB")
        # img.save(file_path)
        # os.remove(file_path)
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
            
print("tyle obrazow jest zlych: ", count_bad)
print("Byl 1 nie w rgb i zostal usuniety")
print("tyle obrazow nie jest w rgb: ", count_not_rgb)

#------------------------------------------------------------------SPRAWDZENIE CZY WSZYSTKIE OBRAZY SA GIT




#USUNIECIE OUTLIEROW-----------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------USUNIECIE OUTLIEROW



#GENERATORY-----------------------------------------------------------------------------
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

validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df,
    x_col = "filename",
    y_col = "class",
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_heigth),
    batch_size = batch_size,
    class_mode = "categorical"
)

#----------------------------------------------------------------------------GENERATORY

print()
print()


#ZEBRANIE INFORMACJI O DANYCH------------------------------------------------------------
data = []

print("Train dataset:")
print("wszystki: ", train_generator.samples)
data.append(["All", train_generator.samples, "Train"])

classes = os.listdir(train_dir)
for class_name in classes:
    class_dir = os.path.join(train_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class {class_name}: {num_images} images")
    data.append([class_name, num_images, "Train"])

print()

print("Validation dataset:")
print("wszystkie: ", validation_generator.samples)
data.append(["All", validation_generator.samples, "Validation"])

classes = os.listdir(val_dir)
for class_name in classes:
    class_dir = os.path.join(val_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class {class_name}: {num_images} images")
    data.append([class_name, num_images, "Validation"])

print()

print("Test dataset:")
print("wszystkie: ", test_generator.samples)
data.append(["All", test_generator.samples, "Test"])

classes = os.listdir(test_dir)
for class_name in classes:
    class_dir = os.path.join(test_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Class {class_name}: {num_images} images")
    data.append([class_name, num_images, "Test"])
#------------------------------------------------------------ZEBRANIE INFORMACJI O DANYCH


print()
print()


#PREZENTACJA DANYCH----------------------------------------------------------------------------------
data_to_graph = pd.DataFrame(data, columns=["Class", "Count", "Dataset"])

pivot_df = data_to_graph.pivot(index='Dataset', columns='Class', values='Count')

pivot_df.plot(kind='bar', figsize=(15, 8))

plt.title('Liczba obrazów w zbiorach danych dla różnych klas')
plt.xlabel('Zbiór danych')
plt.ylabel('Liczba obrazów')

plt.legend(title='Klasa')

# plt.show()

#---------------------------------------------------PREZENTACJA DANYCH


#ZAJMOWANIE SIE OUTLIERAMI-----------------------------------------------------------------------------

def check_image_sizes(directory, expected_size=(150, 150)):
    size_list = np.empty((0, 2))
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    size_list = np.append(size_list, [[img.size[0], img.size[1]]], axis = 0)
    return size_list

size_list = check_image_sizes('cloud_dataset/train')


# print(f"Rozmiar wszystkich {size_list.shape}")

# print(f"srednia {np.mean(size_list, axis=0)}")
# print(f"minimum {np.min(size_list, axis=0)}")
# print(f"maksimum {np.max(size_list, axis=0)}")


plt.figure(figsize=(12,6))
# Histogram dla wysokości
plt.subplot(2, 2, 1)  # 1 wiersz, 2 kolumny, pierwszy wykres
plt.hist(size_list[:, 1], bins=35, color='blue', edgecolor='black')
plt.title('Rozkład wysokości')
plt.xlabel('Wysokość')
plt.ylabel('Liczba próbek')


# Histogram dla szerokości
plt.subplot(2, 2, 2)  # 1 wiersz, 2 kolumny, drugi wykres
plt.hist(size_list[:, 0], bins=35, color='green', edgecolor='black')
plt.title('Rozkład szerokości')
plt.xlabel('Szerokość')
plt.ylabel('Liczba próbek')


width_condition_max = size_list[:, 0] < 700
height_condition_max = size_list[:, 1] < 550
conditions_max = width_condition_max & height_condition_max

width_condition_min = size_list[:, 0] > 100
height_condition_min = size_list[:, 1] > 100
conditions_min = width_condition_min & height_condition_min

conditions = conditions_max & conditions_min

not_outlayers = size_list[conditions]
print("nie outlierzt shape: ", not_outlayers.shape)


plt.subplot(2, 2, 3)
plt.hist(not_outlayers[:, 1], bins=35, color="purple")
plt.title("Nie outlayerzy wysokosc")

plt.subplot(2, 2, 4)
plt.hist(not_outlayers[:, 0], bins=35, color="purple")
plt.title("Nie outlayerzy szerokosc")
plt.tight_layout()

plt.close("all")
#------------------------------------------------------------------------------------------ZAJMOWANIE SIE OUTLIERAMI




#SPRAWDZENIE CZY DOBRZE DZIALA--------------------------------------------------------
images, labels = next(train_generator)
# print("Etykiety w batchu ", labels)

plt.figure(figsize=(10, 10))
for i in range(9):  # Wyświetl 9 obrazów
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Klasa: {labels[i]}")
    plt.axis("off")

plt.show()

print(f"Min: {images.min()}, Max: {images.max()}")
#-----------------------------------------------------------------------SPRAWDZENIE CZY DOBRZE DZIALA
