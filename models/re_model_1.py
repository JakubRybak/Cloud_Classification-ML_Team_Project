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
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l2
from keras_tuner import RandomSearch
from keras.applications import EfficientNetB0, MobileNetV2
from keras.models import Model
from keras import mixed_precision
import datetime

raw_train_dir = "raw_cloud_dataset/train"
raw_test_dir = "raw_cloud_dataset/test"
raw_val_dir = "raw_cloud_dataset/val"

processed_train_dir = "cloud_dataset/train"
processed_val_dir = "cloud_dataset/val"

img_width, img_height = 224, 224
batch_size = 16

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Konfiguracja GPU - dynamiczne alokowanie pamięci + limit bezpieczeństwa
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 1. Włącz dynamiczne alokowanie (zapobiega blokowaniu całej pamięci)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        #To raczej słabo działa
        # # 2. Opcjonalny limit (np. 90% dostępnej pamięci)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]  # 4GB VRAM - bufor
        # )
    except RuntimeError as e:
        print(e)

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


validation_datagen = ImageDataGenerator(
    rescale = 1./255
)

validation_generator = validation_datagen.flow_from_directory(
    processed_val_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical"
)
print()
# ---------------------------------------------------------------------------------------------------

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
# Custom nerual network
# Dobrze sobie radzi 80-90%
# ---------------------------------------------------------------------------------------------------

try:
    model = load_model('best_model.h5')  # Twój zapisany model
    for layer in model.layers:
        layer._name = f"loaded_{layer.name}"
    model = Sequential(model.layers[:-1])
except:
    print("Błąd ładowania modelu. Tworzenie nowego...")
    model = None

for layer in model.layers[:-10]:
    layer.trainable = False
    

 # Dodaj nowe warstwy
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='new_dense'))
model.add(Dropout(0.3, name='new_dropout'))
model.add(Dense(3, activation='softmax', name='new_output'))  # Zakładam, że masz 3 klasy

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# 5. Trening z pełnym monitoringiem
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

# print(f"\nAby uruchomić TensorBoard, wykonaj w terminalu:")
# print(f"tensorboard --logdir {log_dir}")
# print("Następnie otwórz w przeglądarce: http://localhost:6006")
# ---------------------------------------------------------------------------------------------------



