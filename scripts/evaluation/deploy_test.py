import requests
import os

## Ejecutando este script devuelve la clase predicha y la lista de probablilidades en eun JSON

model_url = "https://mamldeploy-production.up.railway.app/predict-land-use/" # Agregue acá la url de railway

# Ruta al archivo de imagen que deseas enviar
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/images_converted/buildings/buildings06.jpg"))

# Abre el archivo de imagen y haz la solicitud POST
with open(file_path, "rb") as img_file:
    files = {"file": ("image.jpeg", img_file, "image/jpeg")}
    response = requests.post(model_url, files=files)

# Verifica si la respuesta fue exitosa
print(f"Status code: {response.status_code}")
print("Response text:", response.text)

# Intentar parsear la respuesta como JSON si es válida
try:
    print(response.json())
except ValueError:
    print("La respuesta no es un JSON válido.")
    
## Ejecutando este script devuelve la clase predicha y la lista de probablilidades en eun JSON