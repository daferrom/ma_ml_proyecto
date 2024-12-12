from tensorflow.keras.layers import Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import tensorflow as tf

## Definición de la arquitectura de EfficientNEtB= para entrenamiento con transfer learning

def train_efficientnetB0_with_transfer_learning(input_shape , num_classes, train_dataset, val_dataset):
    # Definir explícitamente la entrada
    input_layer = Input(shape=input_shape)

    # Cargar EfficientNetB0 con pesos preentrenados y sin la parte superior
    base_model = EfficientNetB0(input_shape=input_shape,
                                include_top=False,
                                weights='imagenet')

    # Congelar las capas de EfficientNetB0 para conservar los pesos
    base_model.trainable = False

    # Crear la parte superior del modelo
    x = base_model(input_layer)  # Asegurarnos de que pasamos la entrada al modelo base
    x = layers.GlobalAveragePooling2D()(x)  # Reduce las dimensiones a un vector
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)  # `num_classes` es el número de clases

    # Crear el modelo final
    model_sup = models.Model(inputs=input_layer, outputs=output_layer)

    # Compilar el modelo con un optimizador y función de pérdida adecuados
    model_sup.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    # Resumen del modelo para ver la estructura
    model_sup.summary()

    # Entrenar solo las capas superiores de EfficientNetB0
    history = model_sup.fit(train_dataset, epochs=10, validation_data=val_dataset)

    # Descongelar algunas capas del modelo base
    base_model.trainable = True

    # Congelar las capas de EfficientNetB0 excepto las últimas 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Compilar el modelo nuevamente con una tasa de aprendizaje más baja
    model_sup.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar el modelo descongelado
    history_enB0_tranfer_learning = model_sup.fit(train_dataset, epochs=10, validation_data=val_dataset)

    # Retornar el modelo compilado
    return model_sup , history_enB0_tranfer_learning
