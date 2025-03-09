from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis


# FUNKCJE
def image_to_rgb(img):
    piksele = np.array(list(img.getdata()))
    return piksele[:, 0].reshape(img.size[1], img.size[0]), piksele[:, 1].reshape(img.size[1], img.size[0]), piksele[:, 2].reshape(img.size[1], img.size[0])
def top_10_values(hist, bin_edges):
    top_10_indices = np.argsort(hist)[-10:]
    top_10_indices = top_10_indices[::-1]
    return bin_edges[top_10_indices]
def gradient_from_channel(channel):
    gradient_x = sobel(channel, axis = 0)
    gradient_y = sobel(channel, axis = 1)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_directions = np. arctan2(gradient_y, gradient_x)
    return gradient_x, gradient_y, gradient, gradient_directions


# Względna ścieżka do folderu clear
relative_path_clear = os.path.join("archive", "train", "train", "clear")
relative_path_cloudy = os.path.join("archive", "train", "train", "cloudy")
relative_path_rainy = os.path.join("archive", "train", "train", "rainy")
relative_paths = [relative_path_clear,relative_path_cloudy, relative_path_rainy]

# narazie do takiego rozmiaru zobaczymy co wyjdzie
target_size = (548,490)

# mozna by brac pod uwage glownie gorny srodek zdjecia bo na dole moga byc drzewa a po bokach budynki
# TO JEST TYLKO DO ANALIZY ZEBY ZOBACZYC
list_width = []
list_height = []
obrazy = []


for path in relative_paths:
    for file_name in os.listdir(path):
        
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(path, file_name)


            with Image.open(file_path) as img:
                
                # TO JEST POTRZEBNE TYLKO DO ANALIZY ZEBY ZOBACZYC
                # width, height = img.size
                # list_width.append(width)
                # list_height.append(height)

                img = img.resize(target_size, Image.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # DO ANALIZY
                obrazy.append(img)


                # TUTAJ SIE ZACZYNA TWORZENIE CECH
                red, green, blue = image_to_rgb(img)

                # SREDNIE I ODCHYLENIA
                mean_red, mean_green, mean_blue = red.mean(), green.mean(), blue.mean()
                std_red, std_green, std_blue = red.std(), green.std(), blue.std()
                
                features = np.array([mean_red, mean_green, mean_blue, std_red, std_green, std_blue])

                # WSPOLCZYNNIKI
                ratio_rg_mean, ratio_rb_mean, ratio_gb_mean = mean_red/mean_green, mean_red/mean_blue, mean_green/mean_blue
                ratio_rg_std, ratio_rb_std, ratio_gb_std = std_red/std_green, std_red/std_blue, std_green/std_blue

                features = np.append(features, [ratio_rg_mean, ratio_rb_mean, ratio_gb_mean, ratio_rg_std, ratio_rb_std, ratio_gb_std])

                # HISTOGRAMY
                hist_red, bin_edges_red = np.histogram(red, bins = 256, range = (0,255))
                hist_green, bin_edges_green = np.histogram(green, bins = 256, range = (0,255))
                hist_blue, bin_edges_blue = np.histogram(blue, bins = 256, range = (0,255))

                # 10 NAJCZESTSZYCH KOLOROW
                top_10_values_red = top_10_values(hist_red, bin_edges_red)
                top_10_values_green = top_10_values(hist_green, bin_edges_green)
                top_10_values_blue = top_10_values(hist_blue, bin_edges_blue)

                features = np.append(features, top_10_values_red)
                features = np.append(features, top_10_values_green)
                features = np.append(features, top_10_values_blue)
                
                # ENTROPIA
                entropia_red = entropy(hist_red[hist_red>0], base = 2)
                entropia_green = entropy(hist_green[hist_green>0], base = 2)
                entropia_blue = entropy(hist_blue[hist_blue>0], base = 2)

                features = np.append(features, [entropia_red, entropia_green, entropia_blue])

                # SKOSNOSC
                skewness_red = skew(red.flatten())
                skewness_green = skew(green.flatten())
                skewness_blue = skew(blue.flatten())

                features = np.append(features, [skewness_red, skewness_green, skewness_blue])

                # KURTOZA
                kurtosis_red = kurtosis(red.flatten())
                kurtosis_green = kurtosis(green.flatten())
                kurtosis_blue = kurtosis(blue.flatten())

                features = np.append(features, [kurtosis_red, kurtosis_green, kurtosis_blue])

                gradient_x_red, gradient_y_red, gradient_red, gradient_directions_red = gradient_from_channel(red)
                gradient_x_green, gradient_y_green, gradient_green, gradient_directions_green = gradient_from_channel(green)
                gradient_x_blue, gradient_y_blue, gradient_blue, gradient_directions_blue = gradient_from_channel(blue)

                features = np.append(features, [gradient_x_red.mean(), gradient_y_red.mean(), gradient_red.mean(), gradient_directions_red.mean()])
                features = np.append(features, [gradient_x_green.mean(), gradient_y_green.mean(), gradient_green.mean(), gradient_directions_green.mean()])
                features = np.append(features, [gradient_x_blue.mean(), gradient_y_blue.mean(), gradient_blue.mean(), gradient_directions_blue.mean()])

                features = np.append(features, [gradient_x_red.std(), gradient_y_red.std(), gradient_red.std(), gradient_directions_red.std()])
                features = np.append(features, [gradient_x_green.std(), gradient_y_green.std(), gradient_green.std(), gradient_directions_green.std()])
                features = np.append(features, [gradient_x_blue.std(), gradient_y_blue.std(), gradient_blue.std(), gradient_directions_blue.std()])
                

                print(features.size)
                break

    
        
#DO SPRAWDZANIA WYMIAROW I TAKIE O

# list_height = np.array(list_height)
# list_width = np.array(list_width)
# list_ratio = list_width/list_height
# print("srednia wysokosc: ", list_height.mean())
# print("srednia szerokosc: ", list_width.mean())
# print("sredni stosunek :", list_ratio.mean())
# print("maksymalna wysokosc: ", list_height.max())
# print("minimalna wysokosc: ", list_height.min())
# print("maksymalna szerokosc: ", list_width.max())
# print("minimalna szerokosc: ", list_width.min())
# print(list_ratio.min())
# print(list_ratio.max())


obrazek = obrazy[2]

# FUNKCJA POCZATKOWA

red, green, blue = image_to_rgb(obrazek)

# TO JEST ZROBIONE
################################################################################################################################
#WSZYSTKO TUTAJ MOZNA DAC DO KOLUMN

#Srednie i odchylenia
mean_red, mean_green, mean_blue = red.mean(), green.mean(), blue.mean()

std_red, std_green, std_blue = red.std(), green.std(), blue.std()

#RATIO KOLOROW

ratio_rg_mean, ratio_rb_mean, ratio_gb_mean = mean_red/mean_green, mean_red/mean_blue, mean_green/mean_blue
ratio_rg_std, ratio_rb_std, ratio_gb_std = std_red/std_green, std_red/std_blue, std_green/std_blue

#NAJCZESTSZE KOLORY
hist_red, bin_edges_red = np.histogram(red, bins = 256, range = (0,255))
hist_green, bin_edges_green = np.histogram(green, bins = 256, range = (0,255))
hist_blue, bin_edges_blue = np.histogram(blue, bins = 256, range = (0,255))


top_10_values_red = top_10_values(hist_red, bin_edges_red)
top_10_values_green = top_10_values(hist_green, bin_edges_green)
top_10_values_blue = top_10_values(hist_blue, bin_edges_blue)

####################################################################################################################################

#GRADIENTY

#POLICZYC Z TEGO SREDNIA, ODCHYLENIE, ENTROPIE, SKOSNOSN, KURTOZA
gradient_x_red, gradient_y_red, gradient_red, gradient_directions_red = gradient_from_channel(red)
gradient_x_green, gradient_y_green, gradient_green, gradient_directions_green = gradient_from_channel(green)
gradient_x_blue, gradient_y_blue, gradient_blue, gradient_directions_blue = gradient_from_channel(blue)

#ENTROPIA

entropia_red = entropy(hist_red[hist_red>0], base = 2)
entropia_green = entropy(hist_green[hist_green>0], base = 2)
entropia_blue = entropy(hist_blue[hist_blue>0], base = 2)

#SKOSNOSC 
skewness_red = skew(red.flatten())

# Kurtoza
kurtosis_red = kurtosis(red.flatten())


########################################################################################################



# TRANSOFRMACJA FOURIERA
f_transform_red = np.fft.fft2(red)
print(f_transform_red)
f_tranform_shifted_red = np.fft.fftshift(f_transform_red)
magnitude_spectrum_red = np.log(np.abs(f_tranform_shifted_red))
#MOZNA POLICZYC SREDNIA, ODCHYLENIE, ENTROPIA, SKOSNOSC, KURTOZA

#PASMA ENERGI
total_energy_red = np.sum(magnitude_spectrum_red)
rows, cols = magnitude_spectrum_red.shape
center_x, center_y = rows//2, cols//2
radius_low = int(0.1 * min(rows,cols))
radius_high = int(0.3 * min(rows, cols))
Y, X = np.ogrid[:rows, :cols]
distance_from_center = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
low_frequency_energy = np.sum(magnitude_spectrum_red[distance_from_center <= radius_low]) / total_energy_red
high_frequency_energy = np.sum(magnitude_spectrum_red[distance_from_center >= radius_high]) / total_energy_red
mid_frequency_energy = 1 - (low_frequency_energy + high_frequency_energy)

f_grad_x_red = sobel(magnitude_spectrum_red, axis = 0)
f_grad_y_red = sobel(magnitude_spectrum_red, axis = 1)
f_directions_red = np.arctan2(f_grad_y_red, f_grad_x_red)
f_directions_red_mean = np.mean(f_directions_red)
