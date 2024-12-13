from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
#EfficientNetB0 con los mejores hiperparametros

# Definir la entrada explícitamente
input_layer = layers.Input(shape=(256, 256, 3))

# Cargar EfficientNetB0 preentrenado (sin las capas superiores)
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_layer)

# Congelar las capas del modelo base para evitar que se entrenen
base_model.trainable = False

# Crear la red de salida con las capas adicionales
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(best_dense_units, activation='relu')(x)
x = layers.Dropout(best_dropout_rate)(x)
output_layer = layers.Dense(num_classes, activation='softmax')(x)

# Construir el modelo final
efficientNetB0_with_best_hp = models.Model(inputs=input_layer, outputs=output_layer)

# Compilar el modelo
efficientNetB0_with_best_hp.compile(
    optimizer=Adam(best_learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar el resumen del modelo
efficientNetB0_with_best_hp.summary()