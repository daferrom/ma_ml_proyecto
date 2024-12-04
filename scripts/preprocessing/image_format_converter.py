import os
import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_image, encode_jpeg
import numpy as np
from PIL import Image

""" Este script convierte las imagenes de .tiff a formato .JPEG para ser compatible con 
    keras y tensorflow
"""

# Ruta al directorio de imágenes .tiff
dataset_path = "../../data/raw_data/UCMerced_LandUse/Images"

output_path = '../../data/images_converted'


os.makedirs(output_path, exist_ok=True)

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        # Crear subcarpeta para la clase en el directorio de salida
        output_class_path = os.path.join(output_path, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for image_name in os.listdir(class_path):
            if image_name.endswith('.tif'):
                # Abrir y convertir la imagen
                img = Image.open(os.path.join(class_path, image_name))
                img = img.convert('RGB')  # Convertir a RGB
                # Guardar la imagen como JPG
                img.save(os.path.join(output_class_path, image_name.replace('.tif', '.jpg')), 'JPEG')


