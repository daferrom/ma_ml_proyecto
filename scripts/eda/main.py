import structural_metrics
import visual_metrics

"""
    se requiere que se ejecuten los scripts de data_acquisition (debe estar ubicado en ma_ml_proyecto\scripts\data_acquisition)
        1. fetch_data.py
        2. unzip_data.py 
    Luego, en la carpeta eda ejecutar (debe estar ubicado en ma_ml_proyecto\scripts\eda)
        1. main.py
"""

dataset_path = '..\\..\\data\\raw_data\\UCMerced_LandUse\\Images'

structural_metrics.execute(dataset_path)
visual_metrics.execute(dataset_path)