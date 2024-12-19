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
from tensorflow.keras.preprocessing import image_dataset_from_directory
import json
from sklearn.metrics import f1_score, recall_score, accuracy_score
from scripts.eda.split_data import train_ds, val_ds
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os

output_path = './data/UCMerced_LandUse/Images_converted'

batch_size = 32
img_size = (256, 256)  # Tamaño de las imágenes
root_seed = 42

train_ds = image_dataset_from_directory(
    output_path,
    validation_split=0.3,  # 70% para entrenamiento, 30% para validación y prueba
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_test_ds = image_dataset_from_directory(
    output_path,
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_batches = int(0.5 * val_test_ds.cardinality().numpy())  # 15% validación
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

num_classes = 21

class_names = [
    'agricultural',
    'airplane',
    'baseballdiamond',
    'beach',
    'buildings',
    'chaparral',
    'denseresidential',
    'forest',
    'freeway',
    'golfcourse',
    'harbor',
    'intersection',
    'mediumresidential',
    'mobilehomepark',
    'overpass',
    'parkinglot',
    'river',
    'runway',
    'sparseresidential',
    'storagetanks',
    'tenniscourt'
]

def create_model(base_model_func, input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    base_model = base_model_func(
        weights='imagenet',
        include_top=False,
        input_tensor=input_layer
    )

    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    x = base_model(input_layer)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer='adam',
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
    plt.show()


    print("\nMétricas Detalladas:")
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"  Mejor Accuracy de Validación: {max(history.history['val_accuracy']):.4f}")
        print(f"  Accuracy Final de Validación: {history.history['val_accuracy'][-1]:.4f}")
        print(f"  Mejor Loss de Validación: {min(history.history['val_loss']):.4f}")


        accuracy_std = np.std(history.history['val_accuracy'])
        print(f"  Estabilidad (Desviación estándar de Accuracy): {accuracy_std:.4f}")

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
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[reduce_lr, early_stopping]
        )

        histories[model_name] = history


        model_name =  f'{model_name}_tl_V1.h5'
        history_name = f'{model_name}_tl_V1_history.json'

        base_dir = os.path.abspath(f'src/{model_name}/models')

        model_path = os.path.join(base_dir, model_name)
        model.save(model_path)


        def visualizar_metricas_test(test_loss, test_acc):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
   
            ax1.bar(['Test Loss'], [test_loss], color='red', alpha=0.6)
            ax1.set_title('Pérdida en Test')
            ax1.set_ylabel('Valor de Pérdida')
            ax1.grid(True, alpha=0.3)
            
 
            ax2.bar(['Test Accuracy'], [test_acc], color='blue', alpha=0.6)
            ax2.set_title('Precisión en Test')
            ax2.set_ylabel('Precisión')
            ax2.set_ylim([0, 1])  
            ax2.grid(True, alpha=0.3)

            ax1.text('Test Loss', test_loss, f'{test_loss:.4f}', 
                    ha='center', va='bottom')
            ax2.text('Test Accuracy', test_acc, f'{test_acc:.4f}', 
                    ha='center', va='bottom')
            
            plt.tight_layout()
            

            plt.title(f'{model_name}')
            plt.savefig('test_metrics.png')
            plt.savefig()


        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc}")

        visualizar_metricas_test(test_loss, test_acc)

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(model.predict(images), axis=1)) 

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)


        conf_matrix = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

        disp.plot(cmap=plt.cm.Blues, values_format='d')

        plt.xticks(rotation=90)
        plt.title(f'Matriz de Confusión {model_name}')
        plt.show()

        def compare_test_results(model, test_ds, model_name):

          y_true = []
          y_pred = []
          y_pred_prob = []
          
          for images, labels in test_ds:
              predictions = model.predict(images)
              y_pred_prob.extend(predictions)
              y_pred.extend(np.argmax(predictions, axis=1))
              y_true.extend(labels.numpy())
          
          y_true = np.array(y_true)
          y_pred = np.array(y_pred)
          y_pred_prob = np.array(y_pred_prob)
          
          test_loss, test_acc = model.evaluate(test_ds, verbose=0)
          f1 = f1_score(y_true, y_pred, average='weighted')
          recall = recall_score(y_true, y_pred, average='weighted')
          
          class_accuracy = []
          for i in range(len(class_names)):
              mask = y_true == i
              if np.any(mask):  
                  class_acc = accuracy_score(y_true[mask], y_pred[mask])
                  class_accuracy.append(class_acc)
              else:
                  class_accuracy.append(0.0)

          class_performance = list(zip(class_names, class_accuracy))

          metrics = {
              'Accuracy': test_acc,
              'F1 Score': f1,
              'Recall': recall,
              'Loss': test_loss
          }
          
          plt.figure(figsize=(15, 10))
          

          plt.subplot(2, 2, 1)
          bars = plt.bar(metrics.keys(), metrics.values())
          plt.title(f'Métricas de Evaluación - {model_name}')
          plt.ylim(0, 1.2)
          
   
          for bar in bars:
              height = bar.get_height()
              plt.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.4f}',
                      ha='center', va='bottom')
          

          plt.subplot(2, 2, 2)
          conf_matrix = confusion_matrix(y_true, y_pred)
          sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
          plt.title('Matriz de Confusión')
          plt.xlabel('Predicción')
          plt.ylabel('Valor Real')
          

          plt.subplot(2, 2, 3)
          top_5 = sorted(class_performance, key=lambda x: x[1], reverse=True)[:5]
          plt.bar([x[0] for x in top_5], [x[1] for x in top_5])
          plt.title('Top 5 Clases Mejor Clasificadas')
          plt.xticks(rotation=45)
          plt.ylim(0, 1)
          

          plt.subplot(2, 2, 4)
          bottom_5 = sorted(class_performance, key=lambda x: x[1])[:5]
          plt.bar([x[0] for x in bottom_5], [x[1] for x in bottom_5])
          plt.title('Top 5 Clases Peor Clasificadas')
          plt.xticks(rotation=45)
          plt.ylim(0, 1)
          
          plt.tight_layout()
          plt.savefig(f'test_results_{model_name}.png')
          plt.show()
          
     
          print(f"\nResumen detallado de evaluación - {model_name}")
          print("-" * 50)
          print(f"Accuracy: {test_acc:.4f}")
          print(f"Loss: {test_loss:.4f}")
          print(f"F1 Score: {f1:.4f}")
          print(f"Recall: {recall:.4f}")
          
          return metrics

        compare_test_results(model, test_ds, model_name)

        model.save('efficientnet_transfer_learning_model.h5')


        history_path = os.path.join(base_dir, history_name)

        with open(history_path, 'w') as f:
            json.dump(history.history, f)

        print(f"Modelo guardado en: {model_path}")
        print(f"Historial guardado en: {history_path}")


    plot_model_comparison(histories)
    plot_comprehensive_model_comparison(histories)

    print("\nResultados finales:")
    for model_name, history in histories.items():
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"{model_name}:")
        print(f"  Accuracy de entrenamiento: {final_train_acc:.4f}")
        print(f"  Accuracy de validación: {final_val_acc:.4f}")