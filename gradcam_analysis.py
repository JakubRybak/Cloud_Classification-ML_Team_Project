import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Model
from metrics import WeightedWeatherPenalty
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Parametry obrazu
img_size = (224, 224)  # zmien na odpowiednie jak przy trenowaniu
base_path = 'C:/Users/Mateu/Cloud_Classification-ML_Team_Project/cloud_dataset/val'
classes = ['clear', 'cloudy', 'rainy']

# Ładujemy model
model = load_model('best_model.keras', custom_objects={
                   'WeightedWeatherPenalty': WeightedWeatherPenalty})

# Przygotuj folder na wyniki
output_dir = 'gradcam_results'
os.makedirs(output_dir, exist_ok=True)

# Funkcja Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([
        model.inputs], [
        model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    print("DEBUG grads:", grads)
    if grads is None:
        print("Błąd: Grads są None!")

    grads = tf.cast(grads, tf.float32)
    last_conv_layer_output = tf.cast(last_conv_layer_output, tf.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    print("Max heatmap value:", tf.reduce_max(heatmap).numpy())
    return heatmap.numpy()

# Losujemy obrazy
selected_images = []
for class_name in classes:
    class_dir = os.path.join(base_path, class_name)
    images = os.listdir(class_dir)
    selected = random.sample(images, 1)
    for img_name in selected:
        selected_images.append((class_name, os.path.join(class_dir, img_name)))

# Grad-CAM dla 9 obrazów
counters = {class_name: 1 for class_name in classes}

for class_name, img_path in selected_images:
    # Wczytaj obraz
    img = load_img(img_path, target_size=img_size)
    if img is None:
        print(f"Nie udało się wczytać obrazu: {img_path}")
        continue

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    print(img_array.shape)
    preds = model.predict(img_array)
    print(img_array.dtype)
    predicted_class = np.argmax(preds[0])

    last_conv_layer_name = "conv2d_5"

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # konwertuj obraz na OpenCV format
    img = img_to_array(load_img(img_path, target_size=img_size))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    if heatmap is None:
        print(f"Heatmap dla obrazu {img_path} nie została wygenerowana.")
        continue

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    out_filename = f"{class_name}_{counters[class_name]}.png"
    out_path = os.path.join(output_dir, out_filename)

    cv2.imwrite(out_path, superimposed_img)
    print(f"Zapisano: {out_path}")

    counters[class_name] += 1

print("Gotowe!")