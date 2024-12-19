import keras_tuner as kt
import json
import mlflow
from mlflow import log_metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from efficientnetB0_hpe_arch import build_model_efficientNetB0_hpe
import tensorflow as tf
import os

# Rutas para cargar los conjuntos de datos
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/processed"))
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_ds")
VAL_DIR = os.path.join(OUTPUT_DIR, "val_ds")

# Cargar los conjuntos de datos guardados
train_ds = tf.data.experimental.load(TRAIN_DIR)
val_ds = tf.data.experimental.load(VAL_DIR)

# Verifica que los conjuntos de datos se han cargado correctamente
print(f"Cargado conjunto de entrenamiento: {train_ds}")
print(f"Cargado conjunto de validación: {val_ds}")
print(f"Cargado conjunto de prueba: {test_ds}")


# Verifica que el tamaño de entrada esté configurado correctamente
input_shape = (256, 256, 3)

# Asegúrate de que el dataset tenga las dimensiones correctas
train_ds_resized = train_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))
val_ds_resized = val_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))

# Función de búsqueda de hiperparámetros
def search_hyperparameters():
    # Configurar el tuner
    tuner = kt.RandomSearch(
        build_model_efficientNetB0_hpe,
        objective='val_accuracy',
        max_trials=10,  # Número de combinaciones limitadas
        executions_per_trial=1,
        directory='efficientnet_tuning',
        project_name='EfficientNetB0_optimization'
    )

    # Configurar el EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_efficientnet_model.keras.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    # Registrar el experimento en MLflow
    experiment_name = "EfficientNetB0_Hyperparameter_Optimization"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Hyperparameter_Tuning_Exploration"):
        # Ejecutar la búsqueda de hiperparámetros
        tuner.search(
            train_ds_resized,
            validation_data=val_ds_resized,
            epochs=10,
            callbacks=[early_stopping, checkpoint]
        )

        # Registrar los hiperparámetros y las métricas de todos los intentos
        for trial in tuner.oracle.trials.values():
            trial_hps = trial.hyperparameters.values
            trial_metrics = trial.metrics

            # Loguear los hiperparámetros de este intento
            mlflow.log_params(trial_hps)

            # Loguear las métricas de este intento (por ejemplo, val_accuracy y val_loss)
            for metric_name, metric_value in trial_metrics.items():
                mlflow.log_metric(metric_name, metric_value)


        # Obtener los mejores hiperparámetros
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Mejores hiperparámetros para EfficientNetB0: {best_hps.values}")

         # Loguear los mejores hiperparámetros en MLflow
        mlflow.log_params(best_hps.values)

        # Guardar los mejores hiperparámetros en un archivo JSON
        with open("best_hyperparameters_efficientnetb0.json", "w") as json_file:
            json.dump(best_hps.values, json_file)

# Ejecutar la búsqueda de hiperparámetros
if __name__ == "__main__":
    search_hyperparameters()
