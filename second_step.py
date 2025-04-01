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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l2
from kerastuner import RandomSearch
from keras.applications import EfficientNetB0, MobileNetV2
from keras.models import Model


raw_train_dir = "raw_cloud_dataset/train"
raw_test_dir = "raw_cloud_dataset/test"
raw_val_dir = "raw_cloud_dataset/val"

processed_train_dir = "cloud_dataset/train"
processed_val_dir = "cloud_dataset/val"

img_width, img_height = 224, 224
batch_size = 16




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
    brightness_range=[0.7, 1.3],
    fill_mode = "nearest"
)

train_generator = train_datagen.flow_from_directory(
    processed_train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)


validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(
    processed_val_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)
print()
# ---------------------------------------------------------------------------------------------------




#CHECKING IF EVERYTHING WORKS
# ---------------------------------------------------------------------------------------------------
# images, labels = next(train_generator)

# plt.figure(figsize=(10, 10))
# for i in range(7):  # Wyświetl 8 obrazów
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i])
#     plt.title(f"Klasa: {labels[i]}")
#     plt.axis("off")

# print(f"Train:   Min: {images.min()}, Max: {images.max()}")

# images2, labels2 = next(validation_generator)
# print(f"Validate:   Min: {images2.min()}, Max: {images2.max()}")

# plt.close('all')
# ---------------------------------------------------------------------------------------------------


#CHECKIN GPU
# print("Dostępne urządzenia:", tf.config.list_physical_devices())
# print("TensorFlow używa:", tf.test.gpu_device_name())


import datetime
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# 4. Callbacki
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    # ModelCheckpoint(
    #     'best_model.h5',
    #     monitor='val_accuracy',
    #     save_best_only=True
    # ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    )
]

# FIRST MODEL
# Dobrze sobie radzi 80-90%
# ---------------------------------------------------------------------------------------------------
# model = Sequential([
#     # Warstwa wejściowa
#     Conv2D(32, (3,3), padding='same', input_shape=(img_width, img_height, 3)),
#     BatchNormalization(),
#     Activation('relu'),
#     Conv2D(32, (3,3), padding='same'),  # dodaj kolejną warstwę conv
#     BatchNormalization(),
#     Activation('relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.3),  # zwiększ dropout
    
#     # Warstwa pośrednia 1
#     Conv2D(64, (3,3), padding='same'),
#     BatchNormalization(),
#     Activation('relu'),
#     Conv2D(64, (3,3), padding='same'),  # dodaj kolejną warstwę conv
#     BatchNormalization(),
#     Activation('relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.3),
    
#     # Warstwa pośrednia 2
#     Conv2D(128, (3,3), padding='same'),
#     BatchNormalization(),
#     Activation('relu'),
#     Conv2D(128, (3,3), padding='same'),  # dodaj kolejną warstwę conv
#     BatchNormalization(),
#     Activation('relu'),
#     MaxPooling2D(2,2),
#     Dropout(0.4),  # większy dropout dla wyższych warstw
    
#     # Klasyfikator
#     Flatten(),
#     Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # większa warstwa
#     BatchNormalization(),
#     Dropout(0.6),  # większy dropout
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # dodatkowa warstwa
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(3, activation='softmax')
# ])


# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy', 
#              keras.metrics.Precision(name='precision'),
#              keras.metrics.Recall(name='recall'),
#              keras.metrics.AUC(name='auc')]
# )


# 5. Trening z pełnym monitoringiem
# history = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),  # Automatyczne obliczenie
#     epochs=100,  # EarlyStopping i tak zatrzyma
#     validation_data=validation_generator,
#     callbacks=callbacks,
#     verbose=1
# )


# plt.figure(figsize=(15, 5))

# # Accuracy
# plt.subplot(1, 3, 1)
# plt.plot(history.history['accuracy'], label='Train')
# plt.plot(history.history['val_accuracy'], label='Validation')
# plt.title('Accuracy')
# plt.legend()

# # Precision
# plt.subplot(1, 3, 2)
# plt.plot(history.history['precision'], label='Train')
# plt.plot(history.history['val_precision'], label='Validation')
# plt.title('Precision')
# plt.legend()

# # Recall
# plt.subplot(1, 3, 3)
# plt.plot(history.history['recall'], label='Train')
# plt.plot(history.history['val_recall'], label='Validation')
# plt.title('Recall')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # 6. Raport końcowy
# print("\nNajlepsze wyniki walidacyjne:")
# print(f"• Accuracy: {max(history.history['val_accuracy']):.4f}")
# print(f"• Precision: {max(history.history['val_precision']):.4f}")
# print(f"• Recall: {max(history.history['val_recall']):.4f}")

# print(f"\nAby uruchomić TensorBoard, wykonaj w terminalu:")
# print(f"tensorboard --logdir {log_dir}")
# print("Następnie otwórz w przeglądarce: http://localhost:6006")
# ---------------------------------------------------------------------------------------------------



# DRUGI MODEL EfficientNetB0
# MEGA SLABY
# ---------------------------------------------------------------------------------------------------
# # Tworzenie modelu bazowego EfficientNetB0
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# # Dodanie warstw na wierzchu istniejącego modelu
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)  # Możesz dostosować liczbę neuronów w warstwie Dense
# predictions = Dense(3, activation='softmax')(x)  # num_classes to liczba klas (3 w Twoim przypadku)

# # Kompilacja modelu
# model = Model(inputs=base_model.input, outputs=predictions)

# # Zamrożenie wag wszystkich warstw EfficientNetB0
# for layer in base_model.layers:
#     layer.trainable = False
# for layer in base_model.layers[-80:]:  # Odmrożenie ostatnich 20 warstw
#     layer.trainable = True

# # Kompilacja modelu z nowymi warstwami
# model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),  # Automatyczne obliczenie
#     epochs=100,  # EarlyStopping i tak zatrzyma
#     validation_data=validation_generator,
#     callbacks=callbacks,
#     verbose=1
# )
# ---------------------------------------------------------------------------------------------------


# Tworzenie modelu bazowego MobileNetV2
# Dobrze sobie radzi na treningowym, na walidacyjnym tak o
# ---------------------------------------------------------------------------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Dodanie warstw na wierzchu istniejącego modelu
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Możesz dostosować liczbę neuronów w warstwie Dense
predictions = Dense(3, activation='softmax')(x)  # num_classes to liczba klas (3 w Twoim przypadku)

# Kompilacja modelu
model = Model(inputs=base_model.input, outputs=predictions)

# Zamrożenie wag wszystkich warstw MobileNetV2
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Automatyczne obliczenie
    epochs=50,  # EarlyStopping i tak zatrzyma
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

# Precision
plt.subplot(1, 3, 2)
plt.plot(history.history['precision'], label='Train')
plt.plot(history.history['val_precision'], label='Validation')
plt.title('Precision')
plt.legend()

# Recall
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Train')
plt.plot(history.history['val_recall'], label='Validation')
plt.title('Recall')
plt.legend()

plt.tight_layout()
plt.show()

# 6. Raport końcowy
print("\nNajlepsze wyniki walidacyjne:")
print(f"• Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"• Precision: {max(history.history['val_precision']):.4f}")
print(f"• Recall: {max(history.history['val_recall']):.4f}")

print(f"\nAby uruchomić TensorBoard, wykonaj w terminalu:")
print(f"tensorboard --logdir {log_dir}")
print("Następnie otwórz w przeglądarce: http://localhost:6006")
# ---------------------------------------------------------------------------------------------------
