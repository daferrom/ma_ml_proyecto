import structural_metrics
import visual_metrics
import split_data

"""
    se requiere que se ejecuten los scripts de data_acquisition (debe estar ubicado en ma_ml_proyecto\scripts\data_acquisition)
        1. fetch_data.py
        2. unzip_data.py 
    Luego, en la carpeta eda ejecutar (debe estar ubicado en ma_ml_proyecto\scripts\eda)
        1. main.py
"""

dataset_path = '..\\..\\data\\raw_data\\UCMerced_LandUse\\Images'
"""
    entrada: dataset_path path de la ruta que contiene las imagenes
    salida: 
        - Total de imagenes en el dataset
        - Formato de las imagenes
        - Clases de las imagenes
"""
structural_metrics.execute(dataset_path)

"""
    entrada: dataset_path path de la ruta que contiene las imagenes
    salida: 
        - Total de imagenes en el dataset
        - Imagen de distribución de contrastes y brillos en la carpeta data
        - Imagen de distribución de tonalidades en la carpeta data
"""
visual_metrics.execute(dataset_path)

split_data.execute