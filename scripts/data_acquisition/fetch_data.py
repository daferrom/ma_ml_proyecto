import requests
import os

# URL del archivo
url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"

# Ruta para la data cruda descargada
download_folder = '../../data/raw_data/'

# Crear la carpeta si no existe
os.makedirs(download_folder, exist_ok=True)

filename = 'ucmerced_dataset.zip'

# Ruta completa donde se guardará el zip
file_path = os.path.join(download_folder, filename)

# Realizar la solicitud GET
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    with open(file_path, "wb") as file:
        file.write(response.content)
    print("Descarga del dataaset UCMerced_LandUse completada con éxito.")
else:
    print("Error al intentar descargar el archivo. Código de estado:", response.status_code)
