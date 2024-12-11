from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras import layers
import os

PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/images_converted')

BATCH_SIZE = 32
IMG_SIZE = (256, 256)
ROOT_SEED = 42


 # Carga y partición de datos
train_ds = image_dataset_from_directory(
        PREPROCESSED_DATA_PATH,
        validation_split=0.3,  # 70% para entrenamiento, 30% para validación y prueba
        subset="training",
        seed=ROOT_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

val_test_ds = image_dataset_from_directory(
        PREPROCESSED_DATA_PATH,
        validation_split=0.3,
        subset="validation",
        seed=ROOT_SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

# Dividir val_test_ds en validación y prueba (50% cada uno)

val_batches = int(0.5 * val_test_ds.cardinality().numpy())  # 15% validación
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)


    # Verificar los lotes y archivos en las particiones

"""
    Imprime:
    - Valores de los lotes de los conjuntos de entrenamiento , validación y prueba
"""


print("Batches en el conjunto de entrenamiento:", train_ds.cardinality().numpy()) # 46
print("Batches en el conjunto de validación:", val_ds.cardinality().numpy()) #10
print("Batches en el conjunto de prueba:", test_ds.cardinality().numpy()) #10


num_train = train_ds.cardinality().numpy() * BATCH_SIZE
num_val = val_ds.cardinality().numpy() * BATCH_SIZE
num_test = test_ds.cardinality().numpy() * BATCH_SIZE

num_train += sum(1 for _ in train_ds.unbatch())
num_val += sum(1 for _ in val_ds.unbatch())
num_test += sum(1 for _ in test_ds.unbatch())

"""
    Imprime:
    - cantidades de imagenes en cada partición del conjunto de datos
"""

print(f"Total de imágenes en el conjunto de entrenamiento: {num_train}") #2942
print(f"Total de imágenes en el conjunto de validación: {num_val}") #640
print(f"Total de imágenes en el conjunto de prueba: {num_test}") #630


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

# Aplicar la augmentación al conjunto de entrenamiento
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))