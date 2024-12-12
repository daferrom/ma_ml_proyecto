import os
import json
from ..models import train_efficientnetB0_with_transfer_learning
from scripts.eda.split_data import train_ds, val_ds

model_name =  'efficientnetb0_tl_V1.h5'
history_name = 'efficientnetb0_tl_V1_history.json'

# Define el directorio de destino para guardar los archivos
base_dir = os.path.abspath('src/efficientnetb0/models')
os.makedirs(base_dir, exist_ok=True)

model , history = train_efficientnetB0_with_transfer_learning((256, 256, 3), 21, train_ds, val_ds)

# Guardar el modelo
model_path = os.path.join(base_dir, model_name)
model.save(model_path)

# Guardar el historial
history_path = os.path.join(base_dir, history_name)

with open(history_path, 'w') as f:
    json.dump(history.history, f)

print(f"Modelo guardado en: {model_path}")
print(f"Historial guardado en: {history_path}")

