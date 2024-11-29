# Definición de los datos

## Origen de los datos

- El UC Merced Land Use Dataset es un conjunto de imágenes satelitales de Estados Unidos, creado por la Universidad de California en Merced. Estas imágenes cubren 21 categorías de uso del suelo, como áreas agrícolas, residenciales, comerciales e industriales, entre otras. El dataset incluye un total de 2,100 imágenes, cada una de 256x256 píxeles, con 100 imágenes por clase. Es ampliamente utilizado para tareas de clasificación de imágenes y análisis geoespacial, siendo relevante para estudios de urbanismo y planificación territorial.

Las 21 clases son :

| Clase           | Count |
|--------------------|-------|
| forest             | 100   |
| buildings          | 100   |
| river              | 100   |
| mobilehomepark     | 100   |
| harbor             | 100   |
| golfcourse         | 100   |
| agricultural       | 100   |
| runway             | 100   |
| baseballdiamond    | 100   |
| overpass           | 100   |
| chaparral          | 100   |
| tenniscourt        | 100   |
| intersection       | 100   |
| airplane           | 100   |
| parkinglot         | 100   |
| sparseresidential  | 100   |
| mediumresidential  | 100   |
| denseresidential   | 100   |
| beach              | 100   |
| freeway            | 100   |
| storagetanks       | 100   |

## Especificación de los scripts para la carga de datos

- 1. scripts/data_acquisition/fetch_data.py  Este script es responsable de la descarga  UCMerced Land Use. en formato .zip y utiliza la API de requests para descargar el archivo ZIP desde la URL proporcionada por el sitio web de UC Merced Land Use Dataset. "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"

- 2. scripts/data_acquisition/unzip_data.py  el script descomprime el contenido del archivo ZIP en la carpeta ./data/UCMerced_LandUse/, utilizando el módulo zipfile de Python. este codigo validaque la carpeta de destino exista antes de proceder con la descompresión y maneja errores en caso de que no se encuentre el archivo ZIP.

- 3. scripts/data_adquisition/load_data.py carga las imagenes mostrando un resumenn por clase e imprime un ejemplo de uns muestra de las imagenes en que queda guardado en scripts/data_acquisition/examples_images_printed.png. scripts/data_adquisition/load_data_notebook.py muestra el codigo como notebook ejecutado.


## Referencias a rutas de origen y destino

- [ ] Link: http://weegee.vision.ucmerced.edu/datasets/landuse.html

