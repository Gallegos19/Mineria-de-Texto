#!/usr/bin/env python3
"""
Script para instalar todas las dependencias necesarias para la aplicación
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}:")
        print(f"Código de salida: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Instala todas las dependencias necesarias"""
    print("🚀 Iniciando instalación de dependencias...")
    
    # 1. Instalar dependencias de Python
    if not run_command("pip install -r requirements.txt", "Instalación de paquetes Python"):
        return False
    
    # 2. Instalar modelo de spaCy
    if not run_command("python -m spacy download es_core_news_sm", "Instalación de modelo spaCy"):
        print("⚠️ Advertencia: No se pudo instalar el modelo de spaCy")
        print("La aplicación intentará descargarlo en tiempo de ejecución")
    
    # 3. Descargar recursos de NLTK
    print("\n🔄 Descargando recursos de NLTK...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ Recursos de NLTK descargados exitosamente")
    except Exception as e:
        print(f"⚠️ Advertencia: Error descargando recursos de NLTK: {e}")
    
    print("\n✅ Instalación de dependencias completada!")
    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)