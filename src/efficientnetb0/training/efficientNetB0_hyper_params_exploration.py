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
train_ds = tf.data.Dataset.load(TRAIN_DIR)
val_ds = tf.data.Dataset.load(VAL_DIR)

# Verifica que los conjuntos de datos se han cargado correctamente
print(f"Cargado conjunto de entrenamiento: {train_ds}")
print(f"Cargado conjunto de validación: {val_ds}")


# Verifica que el tamaño de entrada esté configurado correctamente
input_shape = (256, 256, 3)

# Asegúrate de que el dataset tenga las dimensiones correctas
train_ds_resized = train_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))
val_ds_resized = val_ds.map(lambda x, y: (tf.image.resize(x, input_shape[:2]), y))

def log_params_safely(params):
    # Verificar si el parámetro ya existe y si es el mismo valor
    for param, value in params.items():
        try:
            mlflow.log_param(param, value)
        except mlflow.exceptions.MlflowException as e:
            # Si se produce un error, lo ignoramos (parámetro ya registrado)
            print(f"Advertencia: No se puede registrar el parámetro {param} nuevamente con valor {value}")


# Función de búsqueda de hiperparámetros
def search_hyperparameters():
    # Configurar el tuner
    tuner = kt.RandomSearch(
        build_model_efficientNetB0_hpe,
        objective='val_accuracy',
        max_trials=10,  # Número de combinaciones limitadas
        executions_per_trial=1,
        directory='efficientnet_tuning',
        project_name='EfficientNetB0_optimization_03'
    )

    # Configurar el EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        'best_efficientnet_model_V3.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Registrar el experimento en MLflow
    experiment_name = "EfficientNetB0_Hyperparameter_Optimization_03"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Hyperparameter_Tuning_Exploration"):
        # Ejecutar la búsqueda de hiperparámetros
        tuner.search(
            train_ds_resized,
            validation_data=val_ds_resized,
            epochs=10,
            callbacks=[early_stopping, checkpoint]
        )


        for trial in tuner.oracle.trials.values():
            trial_hps = trial.hyperparameters.values
            mlflow.log_params(trial_hps)

        # Iterar sobre las métricas registradas en el trial
        for metric_name in trial.metrics.metrics.keys():
            metric_values = trial.metrics.get_history(name=metric_name)
            for epoch, metric_value in enumerate(metric_values):
                # Registrar la métrica en MLflow
                mlflow.log_metric(f"{metric_name}_epoch_{epoch}", metric_value)


        # # Registrar los hiperparámetros y las métricas de todos los intentos
        # for trial in tuner.oracle.trials.values():
        #     trial_hps = trial.hyperparameters.values
        #     trial_metrics = trial.metrics.get_history()

        #     # Loguear los hiperparámetros de este intento
        #     mlflow.log_params(trial_hps)
            
        #     for metric_name, metric_values in trial_metrics.items():
        #         for epoch, metric_value in enumerate(metric_values):
        #             mlflow.log_metric(f"{metric_name}_epoch_{epoch}", metric_value)

            
            # for metric_name, metric_values in trial_metrics.items():
            #     if metric_values:
            #         for epoch, value in enumerate(metric_values):
            #             mlflow.log_metric(f"{metric_name}_epoch_{epoch}", value)

            # # Loguear las métricas de este intento (por ejemplo, val_accuracy y val_loss)
            # for metric_name, metric_value in trial_metrics.items():
            #     mlflow.log_metric(metric_name, metric_value)


        # Obtener los mejores hiperparámetros
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Mejores hiperparámetros para EfficientNetB0: {best_hps.values}")

         # Loguear los mejores hiperparámetros en MLflow
        log_params_safely(trial_hps)
        # mlflow.log_params(best_hps.values)

        # Guardar los mejores hiperparámetros en un archivo JSON
        with open("best_hyperparameters_efficientnetb0.json", "w") as json_file:
            json.dump(best_hps.values, json_file)

# Ejecutar la búsqueda de hiperparámetros
if __name__ == "__main__":
    search_hyperparameters()
