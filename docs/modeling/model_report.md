# Reporte del Modelo Final

## Resumen Ejecutivo

En este proyecto se evaluaron y compararon múltiples arquitecturas de aprendizaje profundo, incluyendo EfficientNetB0, DenseNet121 y ResNet50, para un problema de clasificación. Si bien todos los modelos mostraron un rendimiento competitivo, EfficientNetB0 destacó por su capacidad para alcanzar un accuracy del 94% tras la implementación de técnicas de transfer learning con ajustes mínimos.

Además, mediante la exploración de hiperparámetros utilizando Random Search, se optimizó aún más su desempeño. Este resultado subraya la eficiencia de EfficientNetB0 en lograr un rendimiento sobresaliente con un enfoque computacionalmente económico y adaptativo.

## Descripción del Problema

El proyecto aborda la clasificación de imágenes satelitales para categorizar el uso del suelo, una tarea esencial para la planificación urbana y la gestión de recursos. Este problema es clave para automatizar procesos que tradicionalmente requieren análisis manual, lo que permite mejorar la eficiencia en proyectos de urbanismo y sostenibilidad.

El objetivo principal es desarrollar un modelo de aprendizaje profundo basado en una red neuronal convolucional (CNN) que sea capaz de identificar de manera precisa las distintas categorías de imágenes aéreas. Para ello, se utilizó la arquitectura EfficientNetB0, empleando técnicas de transfer learning y ajustes específicos mediante fine-tuning, lo que asegura un alto desempeño en el reconocimiento de patrones visuales complejos presentes en imágenes satelitales.

Este enfoque está justificado por la capacidad de EfficientNetB0 para equilibrar precisión y eficiencia computacional, lo que lo convierte en una solución viable tanto para implementaciones en infraestructura limitada como para proyectos a gran escala.

## Descripción del Modelo

El modelo desarrollado para resolver el problema de clasificación de imágenes satelitales es una red neuronal convolucional basada en la arquitectura EfficientNetB0. Este modelo se seleccionó por su equilibrio entre rendimiento y eficiencia computacional, lo que lo hace adecuado para procesar imágenes de alta resolución en un contexto con recursos computacionales limitados.

### Metodología Utilizada

Transfer Learning: Se utilizó EfficientNetB0 con pesos preentrenados en el conjunto de datos ImageNet para aprovechar características visuales genéricas. Las capas base del modelo se congelaron inicialmente para entrenar únicamente las capas superiores específicas del problema.

Fine-Tuning: Posteriormente, se descongelaron las últimas capas del modelo base, ajustando los pesos mediante un aprendizaje con una tasa de aprendizaje más baja. Este proceso permitió al modelo refinar características adaptadas al dominio específico de las imágenes satelitales.

Exploración de Hiperparámetros: Se empleó Random Search para optimizar hiperparámetros clave, como la tasa de aprendizaje, la arquitectura de las capas superiores y el factor de regularización, lo que mejoró significativamente el desempeño del modelo.

### Técnicas Empleadas

Global Average Pooling: Reducción de las dimensiones de salida del modelo base a un vector compacto, facilitando la conexión con las capas densas.

Regularización: Uso de Dropout para prevenir el sobreajuste en las capas superiores.

Optimización: Se utilizó el optimizador Adam con una función de pérdida de entropía cruzada categórica, adecuada para problemas de clasificación multiclase.

En conjunto, estas técnicas permitieron alcanzar un modelo robusto con un accuracy del 94%, destacando su capacidad para clasificar imágenes de manera eficiente y precisa.

## Evaluación del Modelo

El modelo final se evaluó utilizando las métricas de pérdida y accuracy, las cuales son fundamentales para analizar el desempeño en problemas de clasificación. La métrica de pérdida cuantifica la discrepancia entre las predicciones del modelo y los valores reales, mientras que el accuracy mide la proporción de predicciones correctas.

Resultados Obtenidos
Pérdida: El modelo alcanzó un valor final de pérdida de 0.24 en los datos de evaluación, lo que indica una buena capacidad para minimizar errores en las predicciones.

Accuracy: Con un accuracy del 94%, el modelo demostró un excelente desempeño al clasificar correctamente la mayoría de las imágenes satelitales según sus categorías de uso de suelo.

### Interpretación de los Resultados

Los valores obtenidos reflejan la efectividad del enfoque basado en transfer learning y fine-tuning con EfficientNetB0. La baja pérdida sugiere que el modelo generaliza bien a nuevos datos y no está sobreajustado, mientras que el alto accuracy confirma su capacidad para manejar adecuadamente la complejidad del conjunto de datos.

Estos resultados destacan la robustez del modelo final y respaldan su aplicación en la automatización de procesos relacionados con el análisis de imágenes satelitales, optimizando tanto tiempo como recursos en tareas urbanísticas y de planificación territorial.

### Tiempos Computacionales

El entrenamiento y la exploración de hiperparámetros se llevaron a cabo en una máquina local con un procesador Apple M2, logrando tiempos computacionales razonables. La fase de exploración de hiperparámetros, utilizando técnicas como Random Search, se completó en menos de una hora, lo que resalta la eficiencia del modelo y la capacidad del hardware para manejar cargas moderadas de trabajo en aprendizaje profundo.

## Conclusiones y Recomendaciones

El modelo desarrollado basado en EfficientNetB0 demostró ser altamente efectivo para la clasificación de imágenes satelitales por categorías de uso de suelo, alcanzando un accuracy del 94% y una pérdida de 0.24. Estos resultados destacan su capacidad de generalización y precisión, incluso con un enfoque computacionalmente económico que incluye técnicas de transfer learning y fine-tuning.

Puntos Fuertes

Precisión y generalización: El alto nivel de accuracy refleja la solidez del modelo para clasificar correctamente las imágenes.
Eficiencia computacional: Los tiempos de entrenamiento y exploración de hiperparámetros en un procesador Apple M2 fueron razonables, mostrando su practicidad en entornos locales sin acceso a GPU avanzadas.
Flexibilidad del transfer learning: La arquitectura EfficientNetB0 permitió obtener resultados destacables con ajustes mínimos en comparación con modelos más complejos como DenseNet121 y ResNet50.

Limitaciones

Dependencia de los datos: El modelo podría no generalizar adecuadamente si se aplica a un conjunto de datos con distribuciones significativamente diferentes.
Escalabilidad: Aunque el entrenamiento fue eficiente en este caso, el procesamiento de conjuntos de datos significativamente más grandes podría requerir hardware más robusto.

Interpretabilidad: 
Como ocurre con la mayoría de los modelos de aprendizaje profundo, explicar las decisiones tomadas por el modelo puede ser un desafío, lo que puede limitar su aplicación en contextos que requieran transparencia.

Recomendaciones
Ampliar el dataset: Incorporar más imágenes provenientes de diversas regiones podría mejorar la generalización del modelo.

Explorar arquitecturas adicionales: Evaluar modelos híbridos o más avanzados podría ofrecer ligeras mejoras en desempeño o interpretabilidad.

Desplegar el modelo en entornos reales: Implementar el modelo en aplicaciones prácticas de planificación urbana para evaluar su impacto directo en la toma de decisiones.

Optimizar la infraestructura: Para aplicaciones a mayor escala, considerar hardware especializado o servicios en la nube para acelerar el procesamiento. A veces el entorno de COlab incluso Colab Pro no tiene las capacidad para avanzar de manera agil en los tiemposdde ejecución de un poyecto de este

Este modelo tiene un potencial significativo para automatizar procesos urbanísticos y clasificar imágenes satelitales con alta precisión, siendo una herramienta valiosa para aplicaciones en planificación territorial y análisis geoespacial.







