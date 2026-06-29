# Usamos una imagen ligera de Python
FROM python:3.11-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos primero el archivo de requerimientos para aprovechar el caché de Docker
COPY requirements.txt .

# Instalamos las librerías
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Comando para ejecutar tu API
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]