#!/bin/bash

# Script para iniciar la aplicación en EC2
set -e

# Variables
APP_DIR="/home/ubuntu/tu-repo"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/gunicorn"
RUN_DIR="/var/run/gunicorn"

# Crear directorios necesarios
sudo mkdir -p $LOG_DIR
sudo mkdir -p $RUN_DIR
sudo chown ubuntu:ubuntu $LOG_DIR
sudo chown ubuntu:ubuntu $RUN_DIR

# Cambiar al directorio de la aplicación
cd $APP_DIR

# Activar entorno virtual
source $VENV_DIR/bin/activate

# Verificar que las dependencias están instaladas
echo "Verificando dependencias..."
python -c "import torch, transformers, flask, nltk, spacy; print('✅ Todas las dependencias están instaladas')"

# Descargar modelos si no existen
echo "Verificando modelos..."
python -c "
try:
    import spacy
    nlp = spacy.load('es_core_news_sm')
    print('✅ Modelo spaCy disponible')
except:
    print('⚠️ Descargando modelo spaCy...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
"

# Iniciar aplicación con Gunicorn
echo "🚀 Iniciando aplicación..."
exec gunicorn --config gunicorn.conf.py app:app