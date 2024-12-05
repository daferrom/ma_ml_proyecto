import os
import matplotlib.pyplot as plt

def count_total_images(dataset_path):
  total_images = 0
  for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path,class_name)
    if os.path.isdir(class_path):
      total_images += len(os.listdir(class_path))
  return total_images




def get_formats(dataset_path):
    formats = set()
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                formats.add(file_name.split('.')[-1])
    return formats


def calculate_dataset_size(dataset_path):
    total_size = 0
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)  # Convertir a MB



def count_images_per_class(dataset_path):
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_name] = count
    return class_counts

def counter_image_class(dataset_path):
    class_counts = count_images_per_class(dataset_path)
    print("Cantidad de imágenes por clase:", class_counts)


    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel('Clases')
    plt.ylabel('Número de imágenes')
    plt.title('Distribución de imágenes por clase')
    plt.show()


def execute(dataset_path):
    total_images = count_total_images(dataset_path)
    print(f"El dataset tiene {total_images} imágenes")
    formats = get_formats(dataset_path)
    print(f"Los formatos de los archivos en el dataset son: {formats}")
    counter_image_class(dataset_path)
    dataset_size = calculate_dataset_size(dataset_path)
    print(f"El tamaño del dataset es de {dataset_size:.2f} MB.")

