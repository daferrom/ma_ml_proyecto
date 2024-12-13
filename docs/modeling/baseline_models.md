# Reporte del Modelo Baseline

Este documento detalla los resultados del modelo baseline utilizado para la clasificación de imágenes aéreas empleando la arquitectura EfficientNetB0 con técnicas de ajuste fino (fine-tuning). Este informe proporcionará una base para comparar el desempeño de modelos posteriores y evaluar las mejoras potenciales.

## Descripción del modelo

El modelo baseline se basa en la arquitectura preentrenada EfficientNetB0. EfficientNetB0 ha demostrado una alta eficiencia computacional y un excelente rendimiento en tareas de clasificación de imágenes. Se optó por este modelo debido a su capacidad para equilibrar la precisión con los recursos computacionales necesarios.

## Variables de entrada

Imágenes aéreas: Las imágenes de entrada son preprocesadas para unificar su tamaño y formato. El código incluye técnicas de normalización y redimensionamiento para garantizar una entrada consistente al modelo.
Etiquetas de clase: ['runway', 'agricultural', 'golfcourse', 'chaparral', 'beach', 'parkinglot', 'sparseresidential', 'denseresidential', 'harbor', 'forest', 'buildings', 'mobilehomepark', 'overpass', 'river', 'storagetanks', 'mediumresidential', 'freeway', 'intersection', 'baseballdiamond', 'tenniscourt', 'airplane']


## Variable objetivo

La variable objetivo representa la clase a la que pertenece una imagen aérea determinada. El modelo baseline se entrena para predecir la etiqueta de clase correcta para una imagen dada.

## Evaluación del modelo

Para evaluar el rendimiento del modelo baseline, se utilizan las siguientes métricas:


### Métricas de evaluación

Accuracy: La proporción de imágenes clasificadas correctamente.
Loss: Una medida del error promedio del modelo al hacer predicciones. Un valor de loss bajo indica un buen rendimiento del modelo.

### Resultados de evaluación

Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.

## Análisis de los resultados

Descripción de los resultados del modelo baseline, incluyendo fortalezas y debilidades del modelo.

## Conclusiones


La gráfica presentada refleja un análisis detallado del rendimiento de las arquitecturas CNN durante el entrenamiento y validación en la clasificación de imágenes aéreas. **EfficientNetB0** se destaca como el modelo más eficiente, con una precisión consistente en los datos de validación y entrenamiento, lo que indica una excelente capacidad de generalización. **DenseNet121** y **ResNet50**, aunque también muestran un rendimiento sólido, presentan un leve sobreajuste, evidenciado por la brecha en las curvas de pérdida entre los conjuntos de entrenamiento y validación. Esto subraya la necesidad de optimizar estas arquitecturas para mejorar su capacidad de generalización.

El análisis de la métrica de pérdida resalta la eficacia de los modelos para minimizar errores durante el entrenamiento, pero también muestra que **los modelos DenseNet121 y ResNet50 son más susceptibles al sobreajuste**. Esto sugiere que técnicas como regularización, ajuste de hiperparámetros o aumento de datos podrían ser esenciales para mejorar su desempeño en datos de validación. Por otro lado, la estabilidad de **EfficientNetB0** lo posiciona como la mejor opción para tareas que requieren robustez en la clasificación de imágenes aéreas.

Finalmente, las conclusiones subrayan que **la calidad del conjunto de datos y el preprocesamiento son factores clave para el éxito del modelo**. Un conjunto de datos diverso y estrategias como el aumento de datos pueden mejorar la capacidad de los modelos para captar características complejas. Además, realizar un análisis de errores en las predicciones fallidas podría proporcionar información valiosa para refinar aún más los modelos. En general, el análisis sugiere que **EfficientNetB0 es la mejor arquitectura para la tarea**, pero hay margen de mejora en las otras alternativas.


## Referencias

Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
