import subprocess
import sys
import os

def install_dependencies():
    """
    Instala las dependencias necesarias para el sistema de minería de texto
    """
    print("Instalando dependencias...")
    
    # Instalar paquetes desde requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Descargar recursos de NLTK
    print("\nDescargando recursos de NLTK...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Descargar modelo de spaCy para español
    print("\nDescargando modelo de spaCy para español...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    except:
        print("Error descargando el modelo de spaCy. Intentando con pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.1.0/es_core_news_sm-3.1.0.tar.gz"])
    
    print("\n✅ Instalación completada con éxito!")
    print("Puedes ejecutar el servidor con: python app.py")

if __name__ == "__main__":
    install_dependencies()