import os
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg



ds_images_path = '../../data/raw_data/UCMerced_LandUse/Images'

## Conteo de imagenes por clase en el dataset
def count_images_per_class(ds_images_path):
    class_counts = {}
    for class_name in os.listdir(ds_images_path):
        class_path = os.path.join(ds_images_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_name] = count
    return class_counts

class_counts = count_images_per_class(ds_images_path)
print("Cantidad de imágenes por clase:", class_counts)


# OUTPUT: 
# Cantidad de imágenes por clase:
# {'forest': 100,
# 'buildings': 100,
# 'river': 100,
# 'mobilehomepark': 100,
# 'harbor': 100,
# 'golfcourse': 100,
# 'agricultural': 100,
# 'runway': 100,
# 'baseballdiamond': 100,
# 'overpass': 100,
# 'chaparral': 100,
# 'tenniscourt': 100,
# 'intersection': 100,
# 'airplane': 100,
# 'parkinglot': 100,
# 'sparseresidential': 100,
# 'mediumresidential': 100,
# 'denseresidential': 100,
# 'beach': 100,
# 'freeway': 100,
# 'storagetanks': 100}

def show_sample(ds_images_path, num_images=5):
    # Lista de todas las carpetas de classes
    classes = os.listdir(ds_images_path)

    plt.figure(figsize=(30, 10))

    # Seleccionar imágenes aleatorias
    for i in range(num_images):
        img_class = random.choice(classes)  # Elegir una img_class aleatoria
        image = random.choice(os.listdir(os.path.join(ds_images_path, img_class)))  # Elegir imagen aleatoria de una img_class

        img_path = os.path.join(ds_images_path, img_class, image)
        img = mpimg.imread(img_path)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(img_class)  # Título de la img_class de la image
    plt.show()


show_sample(ds_images_path, num_images=10)

# OUTPUT: ver './example_images_printed.py'