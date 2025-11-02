#!/bin/bash

# Script de lanzamiento para TinySpeak
# Uso: ./launch.sh

echo "🚀 Iniciando TinySpeak..."

# Activar entorno virtual
if [ -d ".venv" ]; then
    echo "📦 Activando entorno virtual..."
    source .venv/bin/activate
else
    echo "❌ Error: No se encontró el entorno virtual (.venv)"
    echo "Por favor, ejecuta primero: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Verificar que Streamlit esté instalado
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit no está instalado"
    echo "Instalando dependencias..."
    pip install -r requirements.txt
fi

# Verificar que espeak esté instalado
if ! command -v espeak &> /dev/null; then
    echo "⚠️  Advertencia: espeak no está instalado"
    echo "Para síntesis de voz, instala espeak:"
    echo "  Ubuntu/Debian: sudo apt-get install espeak"
    echo "  macOS: brew install espeak"
fi

# Ejecutar pruebas rápidas
echo "🔍 Verificando componentes..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo "✅ Todos los componentes funcionan correctamente"
    echo "🌐 Iniciando aplicación web..."
    streamlit run app.py
else
    echo "❌ Error en las pruebas. Revisa la configuración."
    exit 1
fi