## EfficientNetB0 con Transfer Learning y exploración de Hiperparámetros
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
import keras_tuner as kt

# Función de construcción de EfficientNetB0 con exploración de hiperparámetros
def build_model_efficientNetB0_hpe(hp):
    inputs = layers.Input(shape=(256, 256, 3))

    base_model = EfficientNetB0(include_top=False,
                                weights='imagenet',
                               input_tensor=inputs)

    base_model.trainable = False  # Congelar las capas convolucionales iniciales

    # Añadir capa de pooling global
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Capa densa con unidades ajustables
    x = layers.Dense(units=hp.Choice('dense_units', [64, 128, 256]), activation='relu')(x)

    # Capa de dropout ajustable
    x = layers.Dropout(rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)

    # Capa de salida
    outputs = layers.Dense(21, activation='softmax')(x)

    # Construir el modelo
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compilar el modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3, 5e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model