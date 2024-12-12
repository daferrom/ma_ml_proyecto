import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define el directorio de destino
base_dir = os.path.abspath('src/efficientnetb0/models')

# Cargar el modelo
model_path = os.path.join(base_dir, 'efficientnetb0_tl_V1.h5')
model = load_model(model_path)

# Cargar el historial
history_path = os.path.join(base_dir, 'efficientnetb0_tl_V1_history.json')
with open(history_path, 'r') as f:
    history_tl = json.load(f)

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_tl['accuracy'], label='Entrenamiento')
plt.plot(history_tl['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión en Entrenamiento y Validación')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_tl['loss'], label='Entrenamiento')
plt.plot(history_tl['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Validación')

plt.show()
