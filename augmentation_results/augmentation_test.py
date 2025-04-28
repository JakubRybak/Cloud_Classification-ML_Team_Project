import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from metrics import WeightedWeatherPenalty

# Parametry
test_dir = 'C:/Users/Mateu/Cloud_Classification-ML_Team_Project/cloud_dataset/val'
img_size = (224, 224)
batch_size = 32

# Ładowanie modelu
model = load_model('best_model.keras', custom_objects={'WeightedWeatherPenalty': WeightedWeatherPenalty})

# Funkcja do tworzenia generatora testowego z wybraną augmentacją
def create_augmented_test_generator(directory, augmentation_params, target_size=(224, 224), batch_size=32, class_mode="categorical"):
    datagen = ImageDataGenerator(rescale=1./255, **augmentation_params)

    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )
    return generator

# Typy augmentacji do testu
augmentation_types = {
    "no_augmentation": {},
    "rotation": {"rotation_range": 30},
    "horizontal_flip": {"horizontal_flip": True},
    "zoom": {"zoom_range": 0.2},
    "contrast": {"brightness_range": [0.7, 1.3]},
}

results = []

# Testowanie każdej augmentacji
for name, params in augmentation_types.items():
    print(f"\nTestujemy augmentację: {name}")

    test_gen = create_augmented_test_generator(
        directory=test_dir,
        augmentation_params=params,
        target_size=img_size,
        batch_size=batch_size
    )

    metrics = model.evaluate(test_gen, verbose=1)
    results.append((name, metrics))

# Podsumowanie wyników
print("\nPodsumowanie wyników:")
for name, metrics in results:
    print(f"{name}: {metrics}")

# Jeśli chcesz, można dodatkowo zapisać wyniki do pliku .csv
import pandas as pd
columns = ["Loss", "Accuracy", "WeightedWeatherPenalty", "Precision", "Recall", "AUC"]
df = pd.DataFrame(
    [r[1] for r in results],
    index=[r[0] for r in results],
    columns=columns
)
df.to_csv("augmentation_results.csv")
print("\nZapisano wyniki do augmentation_results.csv!")