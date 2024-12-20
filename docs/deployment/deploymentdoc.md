# Despliegue de modelos

## Infraestructura

Requerimientos:

Hardware:

Processador Apple M2 Pro o GPU NVIDIA Tesla T4
Memoria RAM recomendada: 32 GB
Memoria RAM mínima: 12GB
Almacenamiento: SSD para guradar datsets y checkpoints del mejor modelo con los mejores hipermparámetros.

Software:

Tensoflow | 2.18.0
Keras ! 3.7.0

Librerias adicionales

Entrenamiento
keras_tuner | 1.4.7
mlflow | 2.19.0
numpy | 2.0.2
pandas | 2.2.3
matplotlib | 3.9.4

Plataforma:

Local: Usar Apple MAc book pro
Cloud: Google colab Pro


- **Nombre del modelo:** best_efficiennet_model_v3

- **Plataforma de despliegue:** 

    Railway
    url API desplegada : "https://mamldeploy-production.up.railway.app/predict-land-use/"

- **Requisitos técnicos:** 

 (lista de requisitos técnicos necesarios para el despliegue, como versión de Python, bibliotecas de terceros, hardware, etc.)

La maquina virtual del depsliegue en railway tiene una especificación de 2 vCPU and 512 MB con una imagen de linux ubuntu en un contenedor de Docker:

Especificación container: https://registry.hub.docker.com/layers/railwayapp/nixpacks/ubuntu-1731369831/images/sha256-d711daa7170065deeb30749a7baf95ed0d65f1a92b8bba6db434f3c198872e04?context=explore
Arquitectura en: lunux/amd64

En la carpetas [scripts de despliegue](../../scripts/deploy_repo/) se encuntre un copia en este repositorio del repositorio especifico para despliegue https://github.com/daferrom/ma_ml_deploy que contiene:

| best_efficientnet_model_V3.keras | modelo grabado en formato keras |
| main.py | script de despliegue que se ejecuta en railway |
| railway.json | configuracion de reailway en formato json |
| requirements.txt | dependencias necesarias para el despliegue del modelo |

 
también estan especificadas en el archivo requirements.txt este archivo sólo tiene como dependencia adicional uvicorn sin especificar para permitir que pip3 el gestor de paqutes de paython resulva potenciales conflictos de dependencias con uvicorn de la mejor mamnera. 
Adicionalmente este archivo requirements.txt,  con excepcion dela dependencia de uvicorn conicide con el requirements.txt del presente proyecto para replicar y garanztizar la compatibilidad del modelo y del script de deploy en producción.

| Python | 3.9.21 |

| Paquete                             | Versión       |
|-------------------------------------|---------------|
| absl-py                             | 2.1.0         |
| alembic                             | 1.14.0        |
| annotated-types                     | 0.7.0         |
| anyio                               | 4.7.0         |
| astunparse                          | 1.6.3         |
| blinker                             | 1.9.0         |
| cachetools                          | 5.5.0         |
| certifi                             | 2024.12.14    |
| charset-normalizer                  | 3.4.0         |
| click                               | 8.1.7         |
| cloudpickle                         | 3.1.0         |
| contourpy                           | 1.3.0         |
| cycler                              | 0.12.1        |
| databricks-sdk                      | 0.39.0        |
| Deprecated                          | 1.2.15        |
| docker                              | 7.1.0         |
| exceptiongroup                      | 1.2.2         |
| fastapi                             | 0.115.6       |
| Flask                               | 3.1.0         |
| flatbuffers                         | 24.3.25       |
| fonttools                           | 4.55.3        |
| gast                                | 0.6.0         |
| gitdb                               | 4.0.11        |
| GitPython                           | 3.1.43        |
| google-auth                         | 2.37.0        |
| google-pasta                        | 0.2.0         |
| graphene                            | 3.4.3         |
| graphql-core                        | 3.2.5         |
| graphql-relay                       | 3.2.0         |
| grpcio                              | 1.68.1        |
| gunicorn                            | 23.0.0        |
| h5py                                | 3.12.1        |
| idna                                | 3.10          |
| importlib_metadata                  | 8.5.0         |
| importlib_resources                 | 6.4.5         |
| itsdangerous                        | 2.2.0         |
| Jinja2                              | 3.1.4         |
| joblib                              | 1.4.2         |
| keras                               | 3.7.0         |
| keras-tuner                         | 1.4.7         |
| kiwisolver                          | 1.4.7         |
| kt-legacy                           | 1.0.5         |
| libclang                            | 18.1.1        |
| Mako                                | 1.3.8         |
| Markdown                            | 3.7           |
| markdown-it-py                      | 3.0.0         |
| MarkupSafe                          | 3.0.2         |
| matplotlib                          | 3.9.4         |
| mdurl                               | 0.1.2         |
| ml-dtypes                           | 0.4.1         |
| mlflow                              | 2.19.0        |
| mlflow-skinny                       | 2.19.0        |
| namex                               | 0.0.8         |
| numpy                               | 2.0.2         |
| opentelemetry-api                   | 1.29.0        |
| opentelemetry-sdk                   | 1.29.0        |
| opentelemetry-semantic-conventions  | 0.50b0        |
| opt_einsum                          | 3.4.0         |
| optree                              | 0.13.1        |
| packaging                           | 24.2          |
| pandas                              | 2.2.3         |
| pillow                              | 11.0.0        |
| protobuf                            | 5.29.2        |
| pyarrow                             | 18.1.0        |
| pyasn1                              | 0.6.1         |
| pyasn1_modules                      | 0.4.1         |
| pydantic                            | 2.10.4        |
| pydantic_core                       | 2.27.2        |
| Pygments                            | 2.18.0        |
| pyparsing                           | 3.2.0         |
| python-dateutil                     | 2.9.0.post0   |
| python-multipart                    | 0.0.20        |
| pytz                                | 2024.2        |
| PyYAML                              | 6.0.2         |
| requests                            | 2.32.3        |
| rich                                | 13.9.4        |
| rsa                                 | 4.9           |
| scikit-learn                        | 1.6.0         |
| scipy                               | 1.13.1        |
| six                                 | 1.17.0        |
| smmap                               | 5.0.1         |
| sniffio                             | 1.3.1         |
| SQLAlchemy                          | 2.0.36        |
| sqlparse                            | 0.5.3         |
| starlette                           | 0.41.3        |
| tensorboard                         | 2.18.0        |
| tensorboard-data-server             | 0.7.2         |
| tensorflow                          | 2.18.0        |
| tensorflow-io-gcs-filesystem        | 0.37.1        |
| termcolor                           | 2.5.0         |
| threadpoolctl                       | 3.5.0         |
| typing_extensions                   | 4.12.2        |
| tzdata                              | 2024.2        |
| urllib3                             | 2.2.3         |
| uvicorn                             |               |
| Werkzeug                            | 3.1.3         |
| wrapt                               | 1.17.0        |
| zipp                                | 3.21.0        |

modelo: best_efficientnet_model_v3 entrenado y guardado en formato .keras compatible con TensorFlow (importante usar Tensorflow 2.18 con keras 3.7.0 )
el proposito es evitar este problema conocido https://stackoverflow.com/questions/79150363/keras-and-tensorflow-not-getting-imported-despite-all-installations-working


- **Requisitos de seguridad:** (lista de requisitos de seguridad necesarios para el despliegue, como autenticación, encriptación de datos, etc.)

Aunque el despliegue actual de nuestro modelo en Railway no cuenta con medidas avanzadas de seguridad, es esencial planificar e implementar estrategias a futuro para garantizar la protección de los datos procesados, el acceso controlado y la resiliencia del sistema. A continuación, se detalla cada punto junto con las tecnologías específicas que podrían ser utilizadas:

### 1. Autenticación y Control de Acceso

Implementar JWT o OAuth 2.0 para controlar el acceso a los endpoints.
Usar herramientas como Auth0 o Okta para gestión de usuarios.

### 2. Encriptación de Datos

Usar HTTPS con SSL/TLS para cifrar la comunicación.

### 3. Gestión de Secretos

Almacenar claves y credenciales en variables de entorno o servicios como AWS Secrets Manager o HashiCorp Vault.

### 4.Protección del Endpoint

Implementar rate limiting con herramientas como Flask-Limiter.
Usar Cloudflare o AWS Shield para protección contra ataques DDoS.

### 5. Monitoreo y Registros

Usar Datadog o Logflare para centralizar registros y monitoreo.
Configurar alertas con Prometheus y Grafana.

### 6. Actualización de Dependencias y Gestión de Vulnerabilidades

Usar Dependabot o Renovate para mantener dependencias actualizadas.
Configurar tests automatizados para validar actualizaciones.

- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)


## Código de despliegue

- **Archivo principal:**

    scripts/deploy/main.py

    archivo de prueva para peticiones: scripts/evaluation/deploy_test.py (ejecutando este codigo y cargando cualquier imagen se tiene una respuesta con la predicción de la clase de la imagen y la impresión en consola de su etiqueta correspondiente)

- **Rutas de acceso a los archivos:** (lista de rutas de acceso a los archivos necesarios para el despliegue)

    Repositorio en github del despliegue: https://github.com/daferrom/ma_ml_deploy
    url API desplegada : "https://mamldeploy-production.up.railway.app/predict-land-use/"
    [scripts de despliegue](../../scripts/deploy_repo/main.py)
    [scripts de prueba del despliegue](../../scripts/evaluation/deploy_test.py)

- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)

RAILWAY_PUBLIC_DOMAIN
RAILWAY_PRIVATE_DOMAIN
RAILWAY_PROJECT_NAME
RAILWAY_ENVIRONMENT_NAME
RAILWAY_SERVICE_NAME
RAILWAY_PROJECT_ID
RAILWAY_ENVIRONMENT_ID
RAILWAY_SERVICE_ID

## Documentación del despliegue

- **Instrucciones de instalación:** 

### 1. Para clonar el repositorio ejecutar en la terminal: :

```bash
git clone https://github.com/daferrom/ma_ml_proyecto.git

cd ma_ml_proyecto
```

### 2. Configurar el entorno virtual de python (3.9.21) Para crear el entorno virtual ejecutar en la terminal:

```bash
python3 -m venv venv
```

### 3. Activar el entorno virtual , ejecutar 

```bash
source venv/bin/activate
```

### 3. Isntalar las dependencias 

```bash
pip3 install -r requirements.txt
```

### 4. Configura las claves y credenciales necesarias en Railway o localmente en un archivo .env.

PORT=3001

### 5. Navega a deploy en el repo de despliegue clonado o a deploy_repo en este
```bash
cd scripts/deploy_repo
```

### 6 Inicia el servidor de la API con uvicorn (Opcional)

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### 7 navegar en otra terminal manteniendo el entorno virtual activado
```bash
cd scripts/evaluate/
```

### 8 Reemplazar la ruta de la imagen en el image_path por una cualquiera local.

### 9 (si se esta usando el servidor local) reemplazar en  model_path = "https://0.0.0.0:3001" en deploy_test.py


- **Instrucciones de configuración:** 

Configurar un solo repositorio para despliegue
Crear un archivo main.py que contenga las rutas de los endpoints del API del modelo a desplegar
Copiar el modelo en formato keras (save model) en la carpeta de deploy
Reemplazar el path eal pmodelo en la linea 12 de main de "load_model()"
Cerificar que requirements.txt del deploy coincida con requirements.txt del repo de creacion ,exploración de hiperparametros y entrenamiento.
Vericar y ajustar la configuracion de railway.json segun necesidades de despliegue.
Actualizar el repositorio remotos con los cambios.
Conectar railway al repositorio remoto.
Desplegar desde railway (este proceso esta automatizado en es plataforma)

- **Instrucciones de uso:** 

Para ejecutar el modelo en la plataforma de despliegue solo de debe correr su build proces y para probarlo se ejecuta el script de scripts/evaluate/deploy_test.py mencionado anteriormente

- **Instrucciones de mantenimiento:** (instrucciones detalladas para mantener el modelo en la plataforma de despliegue)

1. Monitoreo y Diagnóstico

Latencia y rendimiento: Configurar métricas en FastAPI usando herramientas como Prometheus o Grafana para rastrear tiempos de respuesta del modelo.
Precisión del modelo: Establecer evaluaciones periódicas mediante un conjunto de datos de validación para detectar degradación del rendimiento predictivo.
Uso de recursos: Supervisar el consumo de CPU, memoria y GPU (si está habilitada) para ajustar los recursos asignados en Railway.
Costo asociado: Railway ofrece monitoreo básico; para métricas avanzadas, evalúa herramientas externas con suscripción.

2. Reentrenamiento y Actualizaciones

Reentrenamiento:
Usar datos frescos para actualizar el modelo y prevenir el drift.
Ejecutar el proceso en entornos externos (como Google Colab o AWS) para reducir costos en Railway.
Reentrenar cada 3-6 meses dependiendo de la estabilidad del dominio.
Versionamiento:
Implementar una nomenclatura clara para las versiones del modelo (e.g., efficientnet_v1.0) y mantener un registro en un repositorio como Git o MLflow.
Probar nuevas versiones localmente o en un entorno staging antes de desplegarlas en producción.
Costo asociado: Procesamiento en plataformas externas (e.g., GPU en AWS) es más económico que hacerlo en Railway.

3. Gestión de Infraestructura en Railway

Autoscaling: Habilitar escalado dinámico en Railway para manejar picos de tráfico y minimizar costos durante horas de baja demanda.
Optimización de contenedores:
Reducir el tamaño de la imagen Docker eliminando dependencias innecesarias.
Utilizar una base como python:slim y cargar EfficientNet con una librería como TensorFlow o PyTorch.
Resiliencia: Configurar reinicios automáticos para manejar fallos en la aplicación FastAPI.
Costo asociado: El escalado dinámico en Railway incrementa costos, pero permite mayor disponibilidad durante picos de uso.

4. Seguridad

Endpoints protegidos:
Usar autenticación con OAuth2 o claves API para limitar el acceso al servicio.
Cifrar datos de entrada y salida con HTTPS (Railway ofrece soporte para TLS).
Control de acceso:
Restringir modificaciones a despliegues solo a usuarios autorizados mediante las políticas de roles de Railway.
Auditorías y logs:
Implementar un sistema de logging para registrar predicciones, errores y actividades sospechosas.
Usar herramientas como Sentry para detectar excepciones en FastAPI.
Costo asociado: Los certificados TLS están incluidos en Railway, pero herramientas como Sentry pueden tener un costo adicional.

5. Gestión de Costos

Optimización de uso:
Configurar Railway para detener instancias durante horas de inactividad si la demanda es predecible.
Evaluar planes de Railway y ajustar el uso a límites gratuitos o escalones más económicos.
Reducir el almacenamiento:
Comprimir el modelo EfficientNet y utilizar formatos como ONNX para reducir el tamaño y mejorar tiempos de carga.
Limpiar logs antiguos regularmente para ahorrar espacio en disco.
Costo asociado: Optimizar el almacenamiento y el tiempo de ejecución puede reducir gastos significativamente en planes pagos.

6. Actualización y Documentación

Mantenimiento del código FastAPI:
Mantener las dependencias actualizadas para evitar vulnerabilidades (usa herramientas como pip-tools).
Documentar cada cambio en los endpoints y el modelo en un archivo README o en un wiki interno.
Evaluación periódica del modelo: Generar reportes trimestrales que analicen la precisión del modelo y el impacto en los costos.
