
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def train_denseNet121_with_transfer_learning(input_shape , num_classes, train_dataset, val_dataset):
    
    input_layer = Input(shape=input_shape)


    base_model = DenseNet121(
        weights='imagenet', 
        include_top=False,
        input_tensor=input_layer
    )

    # Congelar las capas del modelo base inicialmente
    base_model.trainable = False

    # Crear la parte superior del modelo
    x = base_model(input_layer) 
    x = layers.GlobalAveragePooling2D()(x)  # Reduce dimensiones
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    # Crear el modelo final
    denseNet121_model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compilar el modelo con un optimizador
    denseNet121_model.compile(
        optimizer=Adam(learning_rate=0.001),  # Reduce el learning rate si es necesario
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Resumen del modelo
    denseNet121_model.summary()

    history = denseNet121_model.fit(train_dataset, epochs=20, validation_data=val_dataset)

    # Descongelar algunas capas del modelo base
    base_model.trainable = True
    

    # Congelar las capas de EfficientNetB0 excepto las últimas 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Compilar el modelo nuevamente con una tasa de aprendizaje más baja
    denseNet121_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entrenar el modelo descongelado
    history_denseNet121_tranfer_learning = denseNet121_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    # Retornar el modelo compilado
    return denseNet121_model , history_denseNet121_tranfer_learning