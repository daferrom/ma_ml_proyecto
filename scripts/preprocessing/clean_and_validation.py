import os
import tensorflow as tf

def check_images_by_class(image_folder):
    corrupted_images_by_class = {}

    # Recorre cada subdirectorio que representa una clase
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)

        if os.path.isdir(class_folder):
            corrupted_images = []

            for filename in os.listdir(class_folder):
                filepath = os.path.join(class_folder, filename)

                try:
                    img = tf.keras.preprocessing.image.load_img(filepath)  # Cargar imagen
                except (IOError, SyntaxError) as e:
                    corrupted_images.append(filename)

            if corrupted_images:
                corrupted_images_by_class[class_name] = corrupted_images

    return corrupted_images_by_class

corrupted_by_class = check_images_by_class("../../data/raw_data/UCMerced_LandUse/Images")

if corrupted_by_class:
    for class_name, images in corrupted_by_class.items():
        print(f"Imágenes corruptas en la clase '{class_name}': {images}")
else:
    print("Todas las imágenes son válidas en todas las clases.")
    
## OUTPUT : "Todas las imágenes son válidas en todas las clases."