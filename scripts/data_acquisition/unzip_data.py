import os
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import zipfile

zip_file_path = '../../data/raw_data/ucmerced_dataset.zip'

extract_folder = '../../data/raw_data/'

# Verificar si el archivo ZIP existe
if os.path.exists(zip_file_path):
    # Abrir el archivo ZIP
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extraer todos los archivos en la carpeta de destino
        zip_ref.extractall(extract_folder)
    print(f"Archivo descomprimido en {extract_folder}")
else:
    print(f"El archivo ZIP no se encuentra en la ruta: {zip_file_path}")

