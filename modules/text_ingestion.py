import re
from typing import Dict

class TextIngestion:
    """
    Módulo para la ingesta de texto desde diferentes fuentes.
    Permite entrada manual, desde archivos o URLs.
    """
    
    def __init__(self):
        self.supported_formats = ['txt', 'csv', 'json']
    
    def ingest_manual_text(self, text):
        """
        Ingesta de texto manual directo
        """
        if not isinstance(text, str):
            raise ValueError("El texto debe ser una cadena de caracteres")
        
        return {
            'original_text': text,
            'source': 'manual',
            'length': len(text),
            'word_count': len(text.split())
        }
    
    def ingest_from_file(self, file_path):
        """
        Ingesta de texto desde archivo
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return {
                'original_text': text,
                'source': f'file: {file_path}',
                'length': len(text),
                'word_count': len(text.split())
            }
        except Exception as e:
            raise Exception(f"Error al leer archivo: {str(e)}")
    
    def validate_text(self, text_data):
        """
        Validación básica del texto ingresado
        """
        text = text_data['original_text']
        
        validations = {
            'is_empty': len(text.strip()) == 0,
            'min_length': len(text) >= 10,
            'has_letters': bool(re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', text)),
            'encoding_issues': '�' in text
        }
        
        return validations