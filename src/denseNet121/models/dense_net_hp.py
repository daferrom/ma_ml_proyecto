
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

def train_denseNet121():

    input_shape = (256, 256, 3)
    num_classes = 21

    train_ds_resized = train_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))
    val_ds_resized = val_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))


    def build_model_densenet121_hpe(hp):
        inputs = layers.Input(shape=input_shape)

        base_model = DenseNet121(include_top=False,
                                    weights='imagenet',
                                    input_tensor=inputs)

        base_model.trainable = False 

        x = layers.GlobalAveragePooling2D()(base_model.output)

        x = layers.Dense(units=hp.Choice('dense_units', [64, 128, 256]), activation='relu')(x)

        x = layers.Dropout(rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3, 5e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    tuner = kt.RandomSearch(
        build_model_densenet121_hpe,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='densenet_tuning',
        project_name='DenseNet121_optimization'
    )


    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_densenet_model.keras', monitor='val_accuracy', mode='max', save_best_only=True)
    
    tuner.search(
        train_ds_resized,
        validation_data=val_ds_resized,
        epochs=10,
        callbacks=[early_stopping, checkpoint]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Mejores hiperparámetros: {best_hps.values}")

    best_learning_rate = best_hps.values.get('learning_rate')
    best_dropout_rate = best_hps.values.get('dropout_rate')
    best_dense_units = best_hps.values.get('dense_units')

    input_layer = layers.Input(shape=(256, 256, 3))

    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_layer)

    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(best_dense_units, activation='relu')(x)
    x = layers.Dropout(best_dropout_rate)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    denseNet121_with_best_hp = models.Model(inputs=input_layer, outputs=output_layer)

    denseNet121_with_best_hp.compile(
        optimizer=Adam(best_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Mostrar el resumen del modelo
    denseNet121_with_best_hp.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_densenet121_model_bhp.keras', monitor='val_accuracy', mode='max', save_best_only=True)

    # Entrenar el modelo
    history_denseNet121_bhp = denseNet121_with_best_hp.fit(
        train_ds_resized,
        validation_data=val_ds_resized,
        epochs=20,
        callbacks=[early_stopping, checkpoint],
    )