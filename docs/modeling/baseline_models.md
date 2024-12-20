# Reporte del Modelo Baseline

Este documento detalla los resultados del modelo baseline utilizado para la clasificación de imágenes aéreas empleando la arquitectura EfficientNetB0 con técnicas de ajuste fino (fine-tuning). Este informe proporcionará una base para comparar el desempeño de modelos posteriores y evaluar las mejoras potenciales.

## Descripción del modelo

El modelo baseline se basa en la arquitectura preentrenada EfficientNetB0. EfficientNetB0 ha demostrado una alta eficiencia computacional y un excelente rendimiento en tareas de clasificación de imágenes. Se optó por este modelo debido a su capacidad para equilibrar la precisión con los recursos computacionales necesarios.

## Variables de entrada

Imágenes aéreas: Las imágenes de entrada son preprocesadas para unificar su tamaño y formato. El código incluye técnicas de normalización y redimensionamiento para garantizar una entrada consistente al modelo.
Etiquetas de clase: ['runway', 'agricultural', 'golfcourse', 'chaparral', 'beach', 'parkinglot', 'sparseresidential', 'denseresidential', 'harbor', 'forest', 'buildings', 'mobilehomepark', 'overpass', 'river', 'storagetanks', 'mediumresidential', 'freeway', 'intersection', 'baseballdiamond', 'tenniscourt', 'airplane']


* input_shape: Tamaño de las imágenes de entrada.
* num_classes: Número de clases en la clasificación.
* train_dataset: Conjunto de datos de entrenamiento.
* val_dataset: Conjunto de datos de validación.

## Variable objetivo

La variable objetivo representa la clase a la que pertenece una imagen aérea determinada. El modelo baseline se entrena para predecir la etiqueta de clase correcta para una imagen dada.

## Evaluación del modelo

Para evaluar el rendimiento del modelo baseline, se utilizan las siguientes métricas:


### Métricas de evaluación

Accuracy: La proporción de imágenes clasificadas correctamente.
Loss: Una medida del error promedio del modelo al hacer predicciones. Un valor de loss bajo indica un buen rendimiento del modelo.

### Resultados de evaluación

![Resultado del modelo EfficientNetb0](../../src/test_metrics_efficientenetb0.png "Test - Resultados de EfficientNetB0")
![Resultado del modelo DenseNet121](../../src/test_metrics_densenet121.png "Test - Resultados de DenseNet121")
![Resultado del modelo DenseNet121](../../src/test_metrics_resnet50.png "Test - Resultados de DenseNet121")

## Análisis de los resultados

![Perdida y Precisión EfficientNetb0](../../src/test_result_efficientnetb0.png "Perdida y Precisión EfficientNetb0")
![Resultado del modelo DenseNet121](../../src/test_result_densenet121.png "Perdida y Precisión DenseNet121")
![Resultado del modelo DenseNet121](../../src/test_result_resnet50.png "Perdida y Precisión ResNet50")

| Métrica     | Valor   |
|-------------|---------|
| **Accuracy**| 0.9613  |
| **Loss**    | 0.1971  |

## Análisis de los resultados

El modelo baseline muestra un accuracy de 0.9613 y una pérdida de 0.1971, lo cual indica un buen rendimiento general. El alto valor de accuracy sugiere que el modelo es capaz de clasificar correctamente una gran parte de las muestras, lo que es una fortaleza destacable, especialmente si se considera que se trata de un modelo entrenado con transfer learning, utilizando pesos preentrenados de EfficientNetB0.

Sin embargo, aunque el modelo presenta buenos resultados, hay algunas áreas a considerar para mejorar su rendimiento:

Posible sobreajuste: Aunque el accuracy es alto, la pérdida de 0.1971 indica que el modelo podría beneficiarse de una mayor regularización o más entrenamiento en ciertos casos. La mejora de la generalización podría ser necesaria para asegurar que el modelo no se sobreajuste a los datos de entrenamiento.

Necesidad de ajuste fino: Al usar transfer learning, el modelo comenzó con pesos preentrenados que podrían no estar completamente optimizados para el conjunto de datos específico. Descongelar algunas de las capas más profundas y realizar un ajuste fino adicional podría mejorar aún más los resultados, particularmente para tareas complejas.

Optimización en el entrenamiento: podría ser beneficioso explorar otras combinaciones de hiperparámetros o técnicas adicionales como el data augmentation para mejorar la robustez del modelo.

### Análisis del Modelo EfficientNetB0 para Clasificación de Imágenes Satelitales

Al analizar los resultados detallados del modelo **EfficientNetB0** en la tarea de clasificación de imágenes satelitales, se evidencia un desempeño notable respaldado por métricas sólidas: una precisión (*Accuracy*) del **96.13%**, un **F1-Score** de **96.35%** y un **Recall** de **96.45%**. La consistencia de estos valores resalta la capacidad uniforme del modelo para clasificar de manera efectiva, acompañada de un valor de pérdida (*Loss*) bajo de **0.1971**, lo que sugiere una adecuada generalización.

La **matriz de confusión** destaca que el modelo tiene un desempeño excepcional en ciertas categorías. Por ejemplo, las "Top 5 Clases Mejor Clasificadas" — *agricultural*, *airplane*, *baseballdiamond*, *beach* y *chaparral* — presentan resultados casi perfectos, lo que indica que estas clases poseen características visuales muy distintivas fácilmente identificables por el modelo. Además, la matriz muestra pocas confusiones entre categorías, con la mayoría de las predicciones concentradas en la diagonal principal.

Por otro lado, el análisis de las "Top 5 Clases Peor Clasificadas" revela retos específicos en categorías como *denseresidential*, *mediumresidential*, y estructuras urbanas como *buildings*, *golfcourse* y *mobilehomepark*. Esto puede atribuirse a características visuales similares y patrones arquitectónicos compartidos que complican la diferenciación. A pesar de ello, incluso las categorías con menor desempeño mantienen tasas de precisión superiores al **65%**, lo cual es un logro destacable considerando que se trata de un problema de clasificación multiclase con **21 categorías diferentes**.


## Conclusiones


La gráfica presentada refleja un análisis detallado del rendimiento de las arquitecturas CNN durante el entrenamiento y validación en la clasificación de imágenes aéreas. **EfficientNetB0** se destaca como el modelo más eficiente, con una precisión consistente en los datos de validación y entrenamiento, lo que indica una excelente capacidad de generalización. **DenseNet121** y **ResNet50**, aunque también muestran un rendimiento sólido, presentan un leve sobreajuste, evidenciado por la brecha en las curvas de pérdida entre los conjuntos de entrenamiento y validación. Esto subraya la necesidad de optimizar estas arquitecturas para mejorar su capacidad de generalización.

El análisis de la métrica de pérdida resalta la eficacia de los modelos para minimizar errores durante el entrenamiento, pero también muestra que **los modelos DenseNet121 y ResNet50 son más susceptibles al sobreajuste**. Esto sugiere que técnicas como regularización, ajuste de hiperparámetros o aumento de datos podrían ser esenciales para mejorar su desempeño en datos de validación. Por otro lado, la estabilidad de **EfficientNetB0** lo posiciona como la mejor opción para tareas que requieren robustez en la clasificación de imágenes aéreas.

Finalmente, las conclusiones subrayan que **la calidad del conjunto de datos y el preprocesamiento son factores clave para el éxito del modelo**. Un conjunto de datos diverso y estrategias como el aumento de datos pueden mejorar la capacidad de los modelos para captar características complejas. Además, realizar un análisis de errores en las predicciones fallidas podría proporcionar información valiosa para refinar aún más los modelos. En general, el análisis sugiere que **EfficientNetB0 es la mejor arquitectura para la tarea**, pero hay margen de mejora en las otras alternativas.


Sin embargo, aunque los resultados son prometedores, existen áreas en las que se puede mejorar:

Generalización: Aunque el modelo tiene un buen accuracy, la pérdida relativamente alta sugiere que podría haber espacio para mejorar la capacidad de generalización del modelo, reduciendo el sobreajuste. Estrategias adicionales como data augmentation o un mayor entrenamiento con regularización podrían ayudar a abordar este desafío.

Ajuste fino: Aunque se utilizaron capas congeladas durante el entrenamiento inicial, es posible que la descongelación parcial de capas adicionales y un ajuste fino más exhaustivo mejoren la precisión, especialmente en tareas de clasificación más complejas.

Exploración de hiperparámetros: El rendimiento del modelo podría mejorarse aún más mediante la optimización de hiperparámetros, utilizando técnicas como búsqueda aleatoria o búsqueda en cuadrícula para encontrar la mejor configuración de parámetros para la tarea en cuestión.

En resumen, el modelo baseline proporciona una excelente base para futuras mejoras, y su rendimiento podría optimizarse con técnicas de ajuste fino, regularización adicional y exploración más profunda de los hiperparámetros.


