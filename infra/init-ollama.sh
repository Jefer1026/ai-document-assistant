#!/bin/bash
# Iniciar ollama en segundo plano
/bin/ollama serve &
# Esperar a que el servidor arranque
sleep 5
# Descargar el modelo
echo "Descargando llama3.1..."
/bin/ollama pull llama3.1
# Mantener el contenedor vivo
wait