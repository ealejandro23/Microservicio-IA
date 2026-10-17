# =============================================================================
# DOCKERFILE - MICROSERVICIO DE IA (PYTORCH)
# =============================================================================
# PyTorch CPU (sin CUDA) para mantener la imagen liviana
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo del microservicio
COPY app.py modelo.py metricas.py ./
COPY datos/ ./datos/

# Modelo entrenado (generado por el pipeline)
COPY modelo_entrenado.pkl ./modelo_entrenado.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
