from PIL import Image
import os

image_folder = '../../data/images_converted'# Ruta a la carpeta que contiene las imágenes

target_size = (224, 224)  # Tamaño objetivo 224x224 pixeles las imágenes para EfficientNetB0

for class_folder in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = Image.open(image_path)
            img = img.resize(target_size)
            img.save(image_path)
