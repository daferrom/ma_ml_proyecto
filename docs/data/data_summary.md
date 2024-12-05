# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

En esta sección se presenta un resumen general de los datos. Se describe el número total de observaciones, variables, el tipo de variables, la presencia de valores faltantes y la distribución de las variables.


El UC Merced Land Use Dataset contiene 2.100 imágenes distribuidas uniformmemnte en 21 categorías (100 imágenes por categoria). 

* El formato de las imágenes es TIF (Tagged Image File Format) el cual es un formato de archivo gráfico que se utiliza para almacenar imágenes de alta calidad y se emplea comunmente en sectores como la impresión y cartográfía.   

* El tamaño del dataset es de 418.75 MB

* Resolución de las imágenes 256 X 256

* Tamaño promedio de la imágen 205 KB

* Espacio de color: RGB

* Canal Alpha : No

## Resumen de calidad de los datos

### 1. Valores Faltantes 

No se encontraron valores faltantes ya que todas las imágenes estan presentes en cada una d sus carpetas.

### 2. Valores extremos y duplicados 

El análisis de valores extremos no es aplicable en este contexto debido a que todas las imagenes tienen cararristicas generales similares como por ejemplo resolución y formato.

Se empleo en scripts/find_duplicates para el caso de los Valores duplicados se encontro solo una imagen duplicada airplane02.tif y se eliminó en el preporcesamiento.

### 3. Errores

En esta sección se presenta un resumen de la calidad de los datos. Se describe la cantidad y porcentaje de valores faltantes, valores extremos, errores y duplicados. También se muestran las acciones tomadas para abordar estos problemas.

Se utilizó el scripts/clean_and_validation para detectar posibles imagenes corruptas. Concluyendo que no se detectarón imaágenes corruptas.

#### 4. Acciones ejecutadas

* Fue aplicada una conversión de imágenes de formatos .Tiff a JPEG para mejor compatibilidad con modelos de aprendizaje profundo , especialmente se realizó este proceso para grantizar la compatibilidad con TensorFlow y Keras.

* 




## Variable objetivo

Variable objetivo a estimar : La variable objetivo a estimar es la categoría a la que pertenece cada imagen.Las categorías corresponden a diferentes tipos de uso del suelo: las cuales incluyen 0. forest (bosque)<br>1. buildings (edificios)<br>2. river (río)<br>3. mobilehomepark (parque de casas móviles)<br>4. harbor (puerto)<br>5. golfcourse (campo de golf) <br>6. agricultural (agrícola)<br>7. runway (pista de aterrizaje)<br>8. baseballdiamond (campo de béisbol)<br>9. overpass (paso elevado)<br>10. chaparral (matorral)<br>11. tenniscourt (cancha de tenis)<br>12. intersection (intersección)<br>13. airplane (avión)<br>14. parkinglot (estacionamiento)<br>15. sparseresidential (residencial disperso)<br>16. mediumresidential (residencial medio)<br>17. denseresidential (residencial denso)<br>18. beach (playa)<br>19. freeway (autopista)<br>20. storagetanks (tanques de almacenamiento)
## Variables individuales

En este caso las variables que definen a la variable objetivo corresponderian a la representacion vectoriasl de los pixeles y los colores de cada imagen. es 

## Ranking de variables
TODO:
En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo. Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.

## Relación entre variables explicativas y variable objetivo
TODO:
En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.
