from tensorflow.keras.preprocessing.image import ImageDataGenerator

prepocessed_dataset_path = '../../data/images_converted'

# Crear un generador que normalice las imágenes a [0, 1]
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Crear el flujo de imágenes desde un directorio donde se maneja el preprocesamiento
image_generator = datagen.flow_from_directory(
    prepocessed_dataset_path,
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=32
)

## OUTPUT: Found 2100 images belonging to 21 classes. ##