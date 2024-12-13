import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from scripts.eda.split_data import train_ds, val_ds


num_classes = 21


# Modelo batch normalization

def create_model_batch_normalization(base_model_func, input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    base_model = base_model_func(
        weights='imagenet', 
        include_top=False,
        input_tensor=input_layer
    )
    
    # Congelar capas base inicialmente
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    
    x = base_model(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_model_basic(base_model_func, input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    
    base_model = base_model_func(
        weights='imagenet', 
        include_top=False,
        input_tensor=input_layer
    )
    
    # Congelar capas base inicialmente
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    
    x = base_model(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='sparse_categorical_crossentropy',
        mmetrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def plot_model_comparison(histories):
    plt.figure(figsize=(15, 5))
    

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} - Entrenamiento')
        plt.plot(history.history['val_accuracy'], label=f'{name} - Validación')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epocas')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} - Entrenamiento')
        plt.plot(history.history['val_loss'], label=f'{name} - Validación')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epocas')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig()
    plt.close()


def plot_comprehensive_model_comparison(histories):
    plt.style.use('seaborn')
    

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Análisis Detallado de Modelos', fontsize=16)
    

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    

    axs[0, 0].set_title('Accuracy de Entrenamiento y Validación')
    for i, (name, history) in enumerate(histories.items()):
        axs[0, 0].plot(history.history['accuracy'], label=f'{name} - Entrenamiento', color=colors[i], linestyle='-')
        axs[0, 0].plot(history.history['val_accuracy'], label=f'{name} - Validación', color=colors[i], linestyle='--')
    axs[0, 0].set_xlabel('Época')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()
    

    axs[0, 1].set_title('Loss de Entrenamiento y Validación')
    for i, (name, history) in enumerate(histories.items()):
        axs[0, 1].plot(history.history['loss'], label=f'{name} - Entrenamiento', color=colors[i], linestyle='-')
        axs[0, 1].plot(history.history['val_loss'], label=f'{name} - Validación', color=colors[i], linestyle='--')
    axs[0, 1].set_xlabel('Época')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    

    axs[0, 2].set_title('Mejora de Accuracy')
    for i, (name, history) in enumerate(histories.items()):
  
        accuracy_improvement = np.diff(history.history['val_accuracy'])
        axs[0, 2].plot(accuracy_improvement, label=name, color=colors[i])
    axs[0, 2].set_xlabel('Época')
    axs[0, 2].set_ylabel('Cambio en Accuracy')
    axs[0, 2].legend()

    axs[1, 0].set_title('Distribución de Accuracy Final')
    final_accuracies = [history.history['val_accuracy'][-1] for history in histories.values()]
    model_names = list(histories.keys())
    axs[1, 0].bar(model_names, final_accuracies, color=colors)
    axs[1, 0].set_ylabel('Accuracy Final')
    axs[1, 0].set_ylim(0, 1)
    
    axs[1, 1].set_title('Curva de Aprendizaje')
    for i, (name, history) in enumerate(histories.items()):
        axs[1, 1].plot(
            history.history['accuracy'], 
            history.history['val_accuracy'], 
            label=name, 
            color=colors[i]
        )
    axs[1, 1].set_xlabel('Accuracy de Entrenamiento')
    axs[1, 1].set_ylabel('Accuracy de Validación')
    axs[1, 1].legend()
    

    axs[1, 2].set_title('Variabilidad de Accuracy')
    accuracy_data = [history.history['val_accuracy'] for history in histories.values()]
    axs[1, 2].boxplot(accuracy_data, labels=model_names)
    axs[1, 2].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig()
    plt.close()



    print("\nMétricas Detalladas:")
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"  Mejor Accuracy de Validación: {max(history.history['val_accuracy']):.4f}")
        print(f"  Accuracy Final de Validación: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Mejor Loss de Validación: {min(history.history['val_loss']):.4f}")
        

        accuracy_std = np.std(history.history['val_accuracy'])
        print(f"  Estabilidad (Desviación estándar de Accuracy): {accuracy_std:.4f}")


def compare_models_batch(train_ds, val_ds, num_classes, input_shape):

    models_to_compare = {
        'EfficientNetB0': EfficientNetB0,
        'DenseNet121': DenseNet121,
        'ResNet50': ResNet50
    }
    

    histories = {}
    

    for model_name, base_model_func in models_to_compare.items():
        print(f"\nEntrenando {model_name}...")
        

        model = create_model_batch_normalization(base_model_func, input_shape, num_classes)
        

        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.2,
            patience=3,
            min_lr=0.000005
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=7,
            restore_best_weights=True
        )
        

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[reduce_lr, early_stopping]
        )
        

        histories[model_name] = history

        print(f"\nResultados de métricas adicionales para {model_name}:")
        print(f"  Recall final (validación): {history.history['val_recall'][-1]:.4f}")
        print(f"  Precision final (validación): {history.history['val_precision'][-1]:.4f}")
        f1_score = 2 * (history.history['val_precision'][-1] * history.history['val_recall'][-1]) / \
                   (history.history['val_precision'][-1] + history.history['val_recall'][-1] + 1e-8)
        print(f"  F1-Score final (validación): {f1_score:.4f}")
    

    plot_model_comparison(histories)
    plot_comprehensive_model_comparison(histories)
    

    print("\nResultados finales:")
    for model_name, history in histories.items():
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"{model_name}:")
        print(f"  Accuracy de entrenamiento: {final_train_acc:.4f}")
        print(f"  Accuracy de validación: {final_val_acc:.4f}")


def compare_models_basic(train_ds, val_ds, num_classes, input_shape):

    models_to_compare = {
        'EfficientNetB0': EfficientNetB0,
        'DenseNet121': DenseNet121,
        'ResNet50': ResNet50
    }
    

    histories = {}
    

    for model_name, base_model_func in models_to_compare.items():
        print(f"\nEntrenando {model_name}...")
        

        model = create_model_basic(base_model_func, input_shape, num_classes)
        

        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.2,
            patience=3,
            min_lr=0.000005
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=7,
            restore_best_weights=True
        )
        

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[reduce_lr, early_stopping]
        )
        

        histories[model_name] = history
    

    plot_model_comparison(histories)
    

    print("\nResultados finales:")
    for model_name, history in histories.items():
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"{model_name}:")
        print(f"  Accuracy de entrenamiento: {final_train_acc:.4f}")
        print(f"  Accuracy de validación: {final_val_acc:.4f}")


compare_models_batch(
    train_ds, 
    val_ds, 
    num_classes=num_classes,
    input_shape=(256, 256, 3)
)

compare_models_basic(
    train_ds, 
    val_ds, 
    num_classes=num_classes,
    input_shape=(256, 256, 3)
)
