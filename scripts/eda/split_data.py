import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import os

PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/images_converted')
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (256, 256)
ROOT_SEED = 42

# Rutas para guardar los conjuntos de datos
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_ds")
VAL_DIR = os.path.join(OUTPUT_DIR, "val_ds")
TEST_DIR = os.path.join(OUTPUT_DIR, "test_ds")

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

# Augmentación de datos
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

# Aplicar augmentación al conjunto de entrenamiento
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Guardar los conjuntos de datos
tf.data.experimental.save(train_ds, TRAIN_DIR)
tf.data.experimental.save(val_ds, VAL_DIR)
tf.data.experimental.save(test_ds, TEST_DIR)

print(f"Conjuntos de datos guardados en {OUTPUT_DIR}")
