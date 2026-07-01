#!/bin/bash

export OLLAMA_HOST=0.0.0.0
# Iniciar Ollama en segundo plano

echo "Iniciando Ollama..."
ollama serve &
PID=$!

echo "Esperando que Ollama esté listo en el puerto 11434..."

# Bucle de espera
while ! (echo > /dev/tcp/127.0.0.1/11434) >/dev/null 2>&1; do
  sleep 2
done

# Verificación inteligente del modelo
echo "Verificando si el modelo llama3.1 está instalado..."
if ollama list | grep -q "llama3.1"; then
    echo "✅ El modelo llama3.1 ya está disponible localmente."
else
    echo "⏳ Descargando modelo llama3.1 (esto puede tardar unos minutos)..."
    ollama pull llama3.1
    echo "✅ Modelo descargado correctamente."
fi

# Mantener el proceso principal activo
wait $PID