import requests
import os

## Ejecutando este script devuelve la clase predicha y la lista de probablilidades en eun JSON

model_url = "https://mamldeploy-production.up.railway.app/predict-land-use/" # Agregue acá la url de railway

# Ruta al archivo de imagen que deseas enviar
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/images_converted/buildings/buildings06.jpg"))

def make_image_prediction(file_path, url):
    # Abre el archivo de imagen y haz la solicitud POST
    with open(file_path, "rb") as img_file:
        files = {"file": ("image.jpeg", img_file, "image/jpeg")}
        response = requests.post(url, files=files)

    # Verifica si la respuesta fue exitosa
    print(f"Status code: {response.status_code}")
    print("Response text:", response.text)

    # Intentar parsear la respuesta como JSON si es válida
    try:
        print(response.json())
        return response.json()
    except ValueError:
        print("La respuesta no es un JSON válido.")
        
    
## Ejecutando este script devuelve la clase predicha y la lista de probablilidades en eun JSON
class_names = [
    'agricultural',
    'airplane',
    'baseballdiamond',
    'beach',
    'buildings',
    'chaparral',
    'denseresidential',
    'forest',
    'freeway',
    'golfcourse',
    'harbor',
    'intersection',
    'mediumresidential',
    'mobilehomepark',
    'overpass',
    'parkinglot',
    'river',
    'runway',
    'sparseresidential',
    'storagetanks',
    'tenniscourt'
    ]

if __name__ == "__main__":
    response_json = make_image_prediction(image_path, model_url)
    prediction = response_json["prediction"]
    print(" La clase predicha para la imagen es:", class_names[prediction])

