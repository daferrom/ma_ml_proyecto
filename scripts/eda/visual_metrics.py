import os
import numpy as np
import tifffile as tiff 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv


def load_tif_images(path, dataset_path):
    images = []
    image_names = []
    for folder in os.listdir(path):
        folder_path = os.path.join(dataset_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.tif'):
                img_path = os.path.join(folder_path, file_name)
                image = tiff.imread(img_path) 
                if image.max() > 1:
                    image = image / 255.0
                images.append(image)
                image_names.append(file_name)
    return images, image_names


def compute_brightness_contrast(image):
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    brightness = gray_image.mean()
    contrast = gray_image.std()
    return brightness, contrast


def compute_hue_distribution(image):
    hsv_image = rgb_to_hsv(image[:, :, :3])
    hue_channel = hsv_image[:, :, 0]
    return np.histogram(hue_channel, bins=256, range=(0, 1))[0]


def compute_dominant_colors(image, k=3):
    image_reshaped = image[:, :, :3].reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image_reshaped)
    return kmeans.cluster_centers_


def  generate_dataframe(data, images, image_names):
    for img, name in zip(images, image_names):
        brightness, contrast = compute_brightness_contrast(img)
        dominant_colors = compute_dominant_colors(img, k=3)
        data.append({
            "image_name": name,
            "brightness": brightness,
            "contrast": contrast,
            "dominant_colors": dominant_colors
        })


    df = pd.DataFrame(data)
    print(df.head())


    plt.hist(df["brightness"], bins=20, color='orange', alpha=0.7, label='Brillo')
    plt.hist(df["contrast"], bins=20, color='blue', alpha=0.5, label='Contraste')
    plt.title("Distribución de Brillo y Contraste")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()

"""
    Distribución de tonalidades (Hue)
"""
def tones_distribution(images):
    hue_distributions = np.sum(
        [compute_hue_distribution(img) for img in images],
        axis=0
    )

    plt.plot(hue_distributions, color='orange')
    plt.title("Distribución de Tonalidades (Hue) - Dataset Completo")
    plt.xlabel("Tonalidad")
    plt.ylabel("Frecuencia")
    plt.show()


def execute(dataset_path):
    data = []
    images, image_names = load_tif_images(dataset_path, dataset_path)
    print(f"Se cargaron {len(images)} imágenes en formato .tif.")   
    generate_dataframe(data, images, image_names)
    tones_distribution(images)
