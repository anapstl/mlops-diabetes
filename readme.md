# MLOps — CHULETA DE EJECUCIÓN

Proyecto: California Housing + MLflow + FastAPI

IMPORTANTE:
- Ejecutar los comandos desde la raíz del proyecto `mlops/`.
- Tener activado el entorno virtual `.venv`.
- Mantener el backend de archivos `mlruns/` para este ejercicio.
- No es necesario instalar ni configurar SQLite.
- Los comandos están pensados para macOS/Linux.

==================================================
1. ACTIVAR EL ENTORNO VIRTUAL
==================================================

```bash
source .venv/bin/activate
```

Comprobar que se utiliza el Python del entorno:

```bash
which python
```

==================================================
2. PERMITIR EL BACKEND DE CARPETAS DE MLFLOW
==================================================

La versión instalada de MLflow considera el backend de archivos (`mlruns/`) en mantenimiento y lo bloquea por defecto.

Para permitirlo durante la sesión actual:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
```

Comprobar:

```bash
echo $MLFLOW_ALLOW_FILE_STORE
```

Debe mostrar `true`.

IMPORTANTE:
- La variable solo permanece activa en esa terminal.
- Si se abre otra terminal, hay que volver a configurarla.
- Si se ejecuta desde el botón de VS Code, puede que no se herede.
- No se ha cambiado el código para utilizar SQLite.

==================================================
3. PIPELINE DE DATOS
==================================================

Ejecutar en orden, cuando sea necesario generar los datos desde cero:

```bash
python src/data/load_path.py
python src/data/validate_data.py
python src/data/preprocess.py
python src/data/feature_eng.py
```

Archivos principales generados:

```text
data/raw/housing_dataset.csv
data/processed/train_v1.csv
data/processed/train_features_v1.csv
```

PECULIARIDADES:
- `validate_data.py` muestra resultados de validación, pero no detiene automáticamente el pipeline ante filas inválidas.
- `preprocess.py` puede sobrescribir `train_v1.csv`.
- `feature_eng.py` NO sobrescribe `train_features_v1.csv`: si ya existe, lanza `FileExistsError`.
- No hay que repetir el pipeline de datos cada vez que se entrena.

==================================================
4. ENTRENAR LOS MODELOS
==================================================

```bash
python src/models/train.py
```

Entrena:
- LinearRegression.
- RandomForestRegressor.

Registra experimentos, métricas y artefactos en MLflow.

PECULIARIDAD DE SEGURIDAD:

En `train.py` se añadió temporalmente:

```python
skops_trusted_types=["sklearn.tree._tree.Tree"],
```

Esto permite guardar/verificar el Random Forest con la configuración de `skops` de nuestra versión de MLflow.

Solo debe utilizarse con modelos generados por nosotros y de procedencia confiable.

ATENCIÓN:
- Volver a ejecutar `train.py` crea nuevas ejecuciones (Runs).
- No hace falta borrar `mlruns/`.
- Entrenar no equivale a registrar el modelo ni a asignarle el alias `champion`.

==================================================
5. REGISTRAR EL MODELO
==================================================

```bash
python src/models/register_model.py
```

Busca la ejecución finalizada con menor RMSE y registra su artefacto en el Model Registry.

Nombre registrado:

```text
california-housing-model
```

PECULIARIDADES:
- Cada ejecución puede crear una nueva versión registrada.
- Registrar una versión NO le asigna automáticamente el alias `champion`.
- El script busca Runs finalizados; no filtra explícitamente solo los de entrenamiento. Si se acumulan Runs de evaluación, conviene revisar cuál selecciona.

==================================================
6. EVALUAR Y ASIGNAR EL ALIAS CHAMPION
==================================================

```bash
python src/models/evaluate_model.py
```

Evalúa la última versión registrada.

Umbrales del código:

```text
RMSE < 0.6
R²   > 0.7
```

Si supera ambos umbrales, asigna:

```text
california-housing-model@champion
```

PECULIARIDADES:
- Si no supera los umbrales, NO asigna el alias.
- El script puede terminar sin excepción aunque no haya coronado ningún modelo.
- No compara necesariamente al aspirante con el campeón anterior.
- El alias puede apuntar a una versión diferente en ejecuciones posteriores.

==================================================
7. LEVANTAR LA INTERFAZ DE MLFLOW
==================================================

En una terminal independiente:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

Abrir:

http://127.0.0.1:5000

Revisar:
- Experimento `california-housing`.
- Runs y métricas.
- Registered Models.
- Modelo `california-housing-model`.
- Versiones registradas.
- Alias `champion`.

Dejar esa terminal abierta mientras se utiliza la interfaz.

==================================================
8. LEVANTAR FASTAPI CON UVICORN
==================================================

En otra terminal, desde la raíz del proyecto:

```bash
MLFLOW_ALLOW_FILE_STORE=true python -m uvicorn src.api.main:app --reload
```

Mensaje esperado:

```text
Application startup complete.
```

Abrir:

http://127.0.0.1:8000/docs

Endpoints disponibles:

```text
GET  /health
GET  /model-info
POST /predict
```

Comprobaciones rápidas:

http://127.0.0.1:8000/health

http://127.0.0.1:8000/model-info

PECULIARIDADES:
- La API carga el modelo `@champion` al arrancar.
- Si se reasigna el alias, hay que reiniciar la API para cargar la nueva versión.
- El endpoint `/predict` exige las 11 características del modelo, incluidas las 3 derivadas.
- No ejecutar `python src/api/main.py` directamente: utiliza imports relativos.
- `--reload` es para desarrollo local, no para producción.

==================================================
9. CORRECCIÓN REALIZADA EN api/model.py
==================================================

El código original construía manualmente esta ruta:

```text
mlruns/<experiment_id>/<run_id>/artifacts/model
```

Pero nuestra versión de MLflow guarda el modelo en una estructura diferente:

```text
mlruns/models/m-.../artifacts/
```

Por eso se sustituyó el bloque de construcción de rutas por:

```python
_state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
```

Con:

```python
MODEL_URI = "models:/california-housing-model@champion"
```

IMPORTANTE:
- Conservar esta corrección.
- No mover ni renombrar manualmente los artefactos.
- Dejar que MLflow resuelva la ubicación del modelo.

==================================================
10. ORDEN NORMAL DE EJECUCIÓN
==================================================

Si los datos ya existen:

TERMINAL 1:

```bash
source .venv/bin/activate
export MLFLOW_ALLOW_FILE_STORE=true

python src/models/train.py
python src/models/register_model.py
python src/models/evaluate_model.py
```

TERMINAL 2 — MLFLOW UI:

```bash
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

TERMINAL 3 — FASTAPI:

```bash
source .venv/bin/activate
MLFLOW_ALLOW_FILE_STORE=true python -m uvicorn src.api.main:app --reload
```

Si ya existe un campeón registrado y no quieres entrenar de nuevo, basta con levantar MLflow UI y FastAPI.

==================================================
11. ERRORES CONOCIDOS
==================================================

ERROR:
"The filesystem tracking backend is in maintenance mode"

CAUSA:
MLflow bloquea el FileStore antiguo.

SOLUCIÓN:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
```

ERROR:
"Untrusted types found: sklearn.tree._tree.Tree"

CAUSA:
`skops` rechaza por defecto un tipo interno de los árboles.

SOLUCIÓN TEMPORAL UTILIZADA EN EL CURSO:
Autorizar únicamente ese tipo en `train.py`, con artefactos propios y confiables.

ERROR:
"Could not load ... @champion"

POSIBLES CAUSAS:
- No existe el alias.
- La API apunta a otro almacén MLflow.
- Falló la carga del artefacto.
- Se construyó una ruta de artefacto incorrecta.

EN NUESTRO CASO:
El modelo y el alias existían; el problema era la ruta manual del artefacto en `api/model.py`.

ERROR:
"Import .model / .schemas could not be resolved"

COMPROBAR:
- Existen `src/api/model.py` y `src/api/schemas.py`.
- Se utiliza el intérprete de `.venv`.
- Se arranca con `python -m uvicorn src.api.main:app --reload`.

==================================================
12. PENDIENTES AL TERMINAR EL CURSO
==================================================

[ ] Dejar de utilizar `MLFLOW_ALLOW_FILE_STORE=true` cuando ya no sea necesario.

[ ] Eliminar o comentar la excepción temporal:

```python
skops_trusted_types=["sklearn.tree._tree.Tree"],
```

[ ] Revisar otras excepciones temporales introducidas durante las prácticas.

[ ] Conservar la corrección de carga del modelo mediante `MODEL_URI`.

[ ] No eliminar el directorio `mlruns/` sin decidir antes si se quieren conservar los experimentos y modelos.

NOTA:
Retirar la excepción de `skops` puede hacer que el entrenamiento vuelva a fallar con la versión actual de las dependencias. Antes de hacerlo, habrá que decidir una solución compatible y segura.

Mlflow Models: http://127.0.0.1:5000/#/models

===================================================

## Despliegue de la API con Docker

### 1. Objetivo

Contenerizar la API desarrollada con **FastAPI**, incluyendo el modelo de Machine Learning registrado en **MLflow**, para poder ejecutar el servicio en un entorno aislado.

El proyecto utiliza:

- Python 3.13.
- FastAPI y Uvicorn.
- MLflow 3.18.0 con almacenamiento local (`FileStore`).
- Un modelo `RandomForestRegressor` serializado mediante `skops`.
- Docker Desktop en macOS.

### 2. Preparación del contenedor

Se creó un `Dockerfile` basado en `python:3.13-slim`.

Su configuración permite:

1. Establecer `/app` como directorio de trabajo.
2. Instalar las dependencias de Python.
3. Configurar `MLFLOW_ALLOW_FILE_STORE=true`, necesario para utilizar el almacenamiento local de MLflow con nuestra versión.
4. Copiar el código de la aplicación desde `src/`.
5. Copiar el registro y los artefactos de MLflow desde `mlruns/`.
6. Exponer el puerto `8000`.
7. Arrancar la API mediante Uvicorn.

También se configuró `.dockerignore` para evitar copiar archivos innecesarios al contexto de construcción, como entornos virtuales, notebooks, cachés y datos que no necesita la API.

**Importante:** no se excluyó `mlruns/`, porque el contenedor necesita los metadatos y los artefactos del modelo.

### 3. Problema encontrado: MLflow no cargaba el modelo

La imagen Docker se construía correctamente, pero al arrancar FastAPI fallaba la carga del modelo registrado:

```text
RuntimeError: Could not load 'models:/california-housing-model@champion'
```

Para obtener el error original se sustituyó el registro de excepciones por:

```python
logger.exception("Failed to load model")
```

Esto permitió identificar el fallo real:

```text
mlflow.exceptions.MlflowException: No such artifact: ''
```

El problema no era la ausencia del alias `champion`, sino la localización de los artefactos del modelo.

### 4. Diagnóstico mediante Docker Desktop

Se creó un contenedor de inspección para explorar los archivos de la imagen sin arrancar FastAPI:

```bash
docker run -d --name mlops-inspeccion --entrypoint sleep mlops-api 3600
```

Desde **Docker Desktop → Containers → Files**, se comprobó que los artefactos del modelo existían dentro del contenedor:

```text
/app/mlruns/models/
└── m-a64620c6a0f345a6bbfdde0533297390/
    └── artifacts/
        ├── MLmodel
        ├── model.skops
        ├── conda.yaml
        ├── python_env.yaml
        └── requirements.txt
```

También se inspeccionó el registro:

```text
/app/mlruns/models/california-housing-model/version-2/meta.yaml
```

Este archivo identificaba el modelo correcto mediante:

```yaml
name: california-housing-model
version: 2
model_id: m-a64620c6a0f345a6bbfdde0533297390
source: models:/m-a64620c6a0f345a6bbfdde0533297390
```

Sin embargo, su campo `storage_location` contenía una ruta absoluta de macOS:

```text
file:///Users/.../mlruns/.../artifacts
```

Dicha ruta no existía dentro del contenedor Linux, donde los archivos se encontraban bajo `/app/mlruns/`.

### 5. Solución: adaptar las rutas de MLflow durante el build

Se añadió al `Dockerfile`, inmediatamente después de `COPY mlruns/ ./mlruns/`, el siguiente bloque:

```dockerfile
# Adaptar las rutas de los artefactos de MLflow al contenedor
RUN python - <<'PY'
from pathlib import Path
import yaml

root = Path("/app/mlruns")

for metadata_file in root.rglob("meta.yaml"):
    metadata = yaml.safe_load(metadata_file.read_text())

    if not isinstance(metadata, dict):
        continue

    for field in ("storage_location", "artifact_location"):
        value = metadata.get(field)

        if isinstance(value, str) and value.startswith("file:///Users/"):
            marker = "/mlruns/"
            if marker in value:
                relative_path = value.split(marker, 1)[1]
                metadata[field] = (root / relative_path).as_uri()
                metadata_file.write_text(
                    yaml.safe_dump(metadata, sort_keys=False)
                )
PY
```

Este código:

- Busca archivos `meta.yaml` dentro de `mlruns/`.
- Detecta rutas absolutas de macOS en los campos `storage_location` y `artifact_location`.
- Las transforma en rutas válidas dentro del contenedor.
- Mantiene intactas las referencias internas de MLflow, como `models:/m-...`.
- **No modifica los archivos originales del proyecto en macOS**, porque se ejecuta durante la construcción de la imagen Docker.

Esta adaptación es específica del proyecto y de su uso de MLflow con almacenamiento local.

### 6. Construcción y ejecución

Desde la raíz del proyecto:

**Construir la imagen:**

```bash
docker build -t mlops-api .
```

**Arrancar el contenedor:**

```bash
docker run --rm -p 8000:8000 mlops-api
```

La API queda disponible en:

```text
http://localhost:8000
```

Documentación interactiva de Swagger:

```text
http://localhost:8000/docs
```

Si ya existe una instancia local de Uvicorn utilizando el puerto `8000`, debe detenerse antes de arrancar el contenedor.

### 7. Validación final

Tras reconstruir la imagen y arrancar el contenedor, se verificó el funcionamiento de los tres endpoints:

| Método | Endpoint | Resultado |
|---|---|---|
| GET | `/health` | Correcto |
| GET | `/model-info` | Correcto |
| POST | `/predict` | Correcto |

La API carga correctamente el modelo registrado en MLflow mediante:

```python
mlflow.pyfunc.load_model(
    "models:/california-housing-model@champion"
)
```

El endpoint `/predict` devuelve predicciones utilizando el modelo cargado dentro del contenedor.

### 8. Conclusiones y consideraciones

**Resultado:** integración funcional de FastAPI, MLflow y Docker.

Aprendizajes principales:

1. La construcción correcta de una imagen Docker no garantiza que la aplicación pueda arrancar.
2. Los errores genéricos pueden ocultar la causa real; `logger.exception()` permite obtener el traceback completo.
3. En MLflow, los metadatos del registro y los archivos físicos del modelo son elementos diferentes.
4. Copiar `mlruns/` a Docker no garantiza la portabilidad si sus metadatos contienen rutas absolutas del sistema original.
5. Es posible adaptar esas rutas durante el build sin alterar el entorno local.
6. La solución utiliza `FileStore` para el entorno educativo. Para despliegues de producción conviene evaluar un backend y un almacenamiento de artefactos adecuados para ese entorno.

**Estado final: API contenerizada y tres endpoints verificados correctamente.**