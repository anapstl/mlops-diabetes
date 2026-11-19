FROM python:3.13-slim
# FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached independently of code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Allow MLflow filesystem tracking backend
ENV MLFLOW_ALLOW_FILE_STORE=true

# Copy application source and the MLflow registry
COPY src/ ./src/
COPY mlruns/ ./mlruns/

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

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]