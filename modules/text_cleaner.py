import re
from typing import Dict, List

class TextCleaner:
    """
    Módulo para la limpieza básica del texto.
    Elimina caracteres especiales, espacios extra, URLs, emails, etc.
    """
    
    def __init__(self):
        # Patrones regex para diferentes tipos de limpieza
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?[0-9]{1,3}[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\;\:\!\?\-\(\)áéíóúÁÉÍÓÚñÑüÜ]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.multiple_punctuation_pattern = re.compile(r'([.!?]){2,}')
    
    def remove_urls(self, text: str) -> str:
        """Elimina URLs del texto"""
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Elimina direcciones de email del texto"""
        return self.email_pattern.sub('', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Elimina números de teléfono del texto"""
        return self.phone_pattern.sub('', text)
    
    def remove_html_tags(self, text: str) -> str:
        """Elimina etiquetas HTML del texto"""
        return self.html_pattern.sub('', text)
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Elimina caracteres especiales manteniendo letras, números y opcionalmente puntuación
        """
        if keep_punctuation:
            # Mantener puntuación básica y caracteres en español
            cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)áéíóúÁÉÍÓÚñÑüÜ]', ' ', text)
        else:
            # Solo mantener letras, números y espacios
            cleaned = re.sub(r'[^\w\s áéíóúÁÉÍÓÚñÑüÜ]', ' ', text)
        
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """Normaliza espacios en blanco múltiples"""
        # Reemplazar múltiples espacios con uno solo
        text = self.multiple_spaces_pattern.sub(' ', text)
        # Eliminar espacios al inicio y final
        return text.strip()
    
    def normalize_punctuation(self, text: str) -> str:
        """Normaliza puntuación repetida"""
        return self.multiple_punctuation_pattern.sub(r'\1', text)
    
    def remove_extra_newlines(self, text: str) -> str:
        """Elimina saltos de línea excesivos"""
        # Reemplazar múltiples saltos de línea con uno solo
        text = re.sub(r'\n+', '\n', text)
        # Reemplazar saltos de línea con espacios para texto continuo
        text = re.sub(r'\n', ' ', text)
        return text
    
    def clean_encoding_issues(self, text: str) -> str:
        """Corrige problemas de codificación comunes"""
        # Diccionario de reemplazos comunes
        encoding_fixes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã¼': 'ü', 'Â': '', 'â€™': "'", 'â€œ': '"',
            'â€': '"', 'â€¦': '...', 'â€"': '-', '�': ''
        }
        
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def basic_clean(self, text: str, options: Dict = None) -> Dict:
        """
        Realiza limpieza básica completa del texto
        
        Args:
            text: Texto a limpiar
            options: Diccionario con opciones de limpieza
        
        Returns:
            Diccionario con texto limpio y estadísticas
        """
        if options is None:
            options = {
                'remove_urls': True,
                'remove_emails': True,
                'remove_phones': True,
                'remove_html': True,
                'remove_special_chars': True,
                'keep_punctuation': True,
                'normalize_whitespace': True,
                'normalize_punctuation': True,
                'remove_newlines': True,
                'fix_encoding': True
            }
        
        original_text = text
        original_length = len(text)
        
        # Aplicar limpiezas según opciones
        if options.get('fix_encoding', True):
            text = self.clean_encoding_issues(text)
        
        if options.get('remove_html', True):
            text = self.remove_html_tags(text)
        
        if options.get('remove_urls', True):
            text = self.remove_urls(text)
        
        if options.get('remove_emails', True):
            text = self.remove_emails(text)
        
        if options.get('remove_phones', True):
            text = self.remove_phone_numbers(text)
        
        if options.get('remove_newlines', True):
            text = self.remove_extra_newlines(text)
        
        if options.get('remove_special_chars', True):
            text = self.remove_special_characters(text, options.get('keep_punctuation', True))
        
        if options.get('normalize_punctuation', True):
            text = self.normalize_punctuation(text)
        
        if options.get('normalize_whitespace', True):
            text = self.normalize_whitespace(text)
        
        # Calcular estadísticas
        cleaned_length = len(text)
        reduction_percentage = ((original_length - cleaned_length) / original_length) * 100 if original_length > 0 else 0
        
        return {
            'original_text': original_text,
            'cleaned_text': text,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'reduction_percentage': round(reduction_percentage, 2),
            'cleaning_options': options
        }