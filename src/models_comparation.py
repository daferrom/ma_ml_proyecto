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

def create_model(base_model_func, input_shape, num_classes):
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

def plot_model_comparison(histories):
    plt.figure(figsize=(15, 5))
    

    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} - Training')
        plt.plot(history.history['val_accuracy'], label=f'{name} - Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} - Training')
        plt.plot(history.history['val_loss'], label=f'{name} - Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def compare_models(train_ds, val_ds, num_classes, input_shape):

    models_to_compare = {
        'EfficientNetB0': EfficientNetB0,
        'DenseNet121': DenseNet121,
        'ResNet50': ResNet50
    }
    

    histories = {}
    

    for model_name, base_model_func in models_to_compare.items():
        print(f"\nEntrenando {model_name}...")
        

        model = create_model(base_model_func, input_shape, num_classes)
        

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


compare_models(
    train_ds, 
    val_ds, 
    num_classes=num_classes,
    input_shape=(256, 256, 3)
)
