#!/bin/bash
# Iniciar el servicio en segundo plano
ollama serve & 

# Esperar a que el puerto 11434 responda usando /dev/tcp
echo "Esperando que Ollama inicie..."

# Esto intenta abrir una conexión TCP al puerto 11434
while ! (echo > /dev/tcp/127.0.0.1/11434) >/dev/null 2>&1; do
  sleep 2
done

echo "Ollama listo. Descargando modelo..."
ollama pull llama3.1

# Mantener el proceso principal activo
wait $!
EOF