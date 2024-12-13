import os
import tensorflow as tf
from tensorflow.keras.models import load_model

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/processed"))
TEST_DIR = os.path.join(OUTPUT_DIR, "test_ds")

test_ds = tf.data.experimental.load(TEST_DIR)


# Define el directorio de destino
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src/efficientnetb0/training"))

# Cargar el modelo
model_path = os.path.join(base_dir, 'best_efficientnet_model.keras.h5')
model = load_model(model_path)

# Evaluar el modelo EfficientNetB0 con mejores hp
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")
