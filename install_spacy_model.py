#!/usr/bin/env python3
"""
Script para instalar el modelo de spaCy necesario para la aplicación
"""
import subprocess
import sys

def install_spacy_model():
    """Instala el modelo de spaCy para español"""
    try:
        print("Instalando modelo de spaCy para español...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "es_core_news_sm"
        ])
        print("✓ Modelo de spaCy instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando modelo de spaCy: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = install_spacy_model()
    sys.exit(0 if success else 1)