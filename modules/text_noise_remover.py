import re
from typing import Dict, List, Set
import spacy
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords.words('spanish')
except:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('es_core_news_sm')
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load('es_core_news_sm')

class NoiseRemover:
    """
    Módulo para la eliminación de ruido del texto.
    Elimina stopwords, palabras muy frecuentes/raras, contenido irrelevante
    y ruido específico del dominio.
    """
    
    def __init__(self):
        # Stopwords básicas en español
        self.spanish_stopwords = set(stopwords.words('spanish'))
        
        # Stopwords adicionales personalizadas
        self.custom_stopwords = {
            # Conectores y muletillas
            'pues', 'bueno', 'entonces', 'así', 'ahora', 'luego', 'después',
            'antes', 'mientras', 'durante', 'mediante', 'según', 'incluso',
            'además', 'también', 'tampoco', 'sino', 'aunque', 'sin embargo',
            'no obstante', 'por tanto', 'por consiguiente', 'en consecuencia',
            
            # Palabras de relleno
            'cosa', 'cosas', 'algo', 'nada', 'todo', 'todos', 'todas',
            'mucho', 'muchos', 'muchas', 'poco', 'pocos', 'pocas',
            'bastante', 'demasiado', 'suficiente',
            
            # Palabras muy generales
            'forma', 'manera', 'modo', 'tipo', 'tipos', 'clase', 'clases',
            'parte', 'partes', 'lado', 'lados', 'vez', 'veces', 'momento',
            'momentos', 'tiempo', 'tiempos', 'lugar', 'lugares', 'caso', 'casos'
        }
        
        # Stopwords específicas para contexto ambiental (palabras muy comunes que no aportan)
        self.environmental_stopwords = {
            'tema', 'temas', 'problema', 'problemas', 'situación', 'situaciones',
            'aspecto', 'aspectos', 'factor', 'factores', 'elemento', 'elementos',
            'punto', 'puntos', 'cuestión', 'cuestiones', 'asunto', 'asuntos'
        }
        
        # Combinar todos los stopwords
        self.all_stopwords = (self.spanish_stopwords | 
                             self.custom_stopwords | 
                             self.environmental_stopwords)
        
        # Patrones de ruido común
        self.noise_patterns = [
            r'\b\w{1,2}\b',  # Palabras muy cortas (1-2 caracteres)
            r'\b\d+\b',      # Números sueltos (opcional)
            r'[^\w\s]',      # Signos de puntuación sueltos
            r'\b[a-zA-Z]\b', # Letras sueltas
        ]
        
        # Palabras de ruido específicas
        self.noise_words = {
            'mm', 'hmm', 'eh', 'ah', 'oh', 'uh', 'um', 'er',
            'ok', 'okay', 'si', 'no', 'ya', 'je', 'ja', 'jaja', 'jeje',
            'etc', 'etcetera', 'bla', 'blah'
        }
    
    def remove_stopwords(self, text: str, custom_stopwords: set = None) -> str:
        """
        Elimina stopwords del texto
        
        Args:
            text: Texto a procesar
            custom_stopwords: Stopwords adicionales personalizadas
        
        Returns:
            Texto sin stopwords
        """
        words = text.split()
        stopwords_to_use = self.all_stopwords.copy()
        
        if custom_stopwords:
            stopwords_to_use.update(custom_stopwords)
        
        filtered_words = [
            word for word in words 
            if word.lower() not in stopwords_to_use
        ]
        
        return ' '.join(filtered_words)
    
    def remove_by_frequency(self, text: str, min_freq: int = 2, max_freq_ratio: float = 0.1) -> Dict:
        """
        Elimina palabras muy raras o muy frecuentes
        
        Args:
            text: Texto a procesar
            min_freq: Frecuencia mínima para mantener una palabra
            max_freq_ratio: Ratio máximo de frecuencia (ej: 0.1 = 10% del total)
        
        Returns:
            Diccionario con texto filtrado y estadísticas
        """
        words = text.split()
        word_freq = {}
        
        # Contar frecuencias
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        total_words = len(words)
        max_freq = int(total_words * max_freq_ratio)
        
        # Identificar palabras a eliminar
        rare_words = {word for word, freq in word_freq.items() if freq < min_freq}
        frequent_words = {word for word, freq in word_freq.items() if freq > max_freq}
        words_to_remove = rare_words | frequent_words
        
        # Filtrar palabras
        filtered_words = [
            word for word in words 
            if word.lower() not in words_to_remove
        ]
        
        return {
            'filtered_text': ' '.join(filtered_words),
            'original_word_count': len(words),
            'filtered_word_count': len(filtered_words),
            'rare_words_removed': len(rare_words),
            'frequent_words_removed': len(frequent_words),
            'rare_words': list(rare_words)[:10],  # Mostrar solo las primeras 10
            'frequent_words': list(frequent_words)
        }
    
    def remove_noise_patterns(self, text: str) -> str:
        """
        Elimina patrones de ruido usando regex
        """
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_noise_words(self, text: str) -> str:
        """
        Elimina palabras de ruido específicas
        """
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower() not in self.noise_words
        ]
        return ' '.join(filtered_words)
    
    def remove_short_words(self, text: str, min_length: int = 3) -> str:
        """
        Elimina palabras muy cortas
        """
        words = text.split()
        filtered_words = [
            word for word in words 
            if len(word) >= min_length
        ]
        return ' '.join(filtered_words)
    
    def remove_non_alphabetic(self, text: str, keep_numbers: bool = False) -> str:
        """
        Elimina tokens que no son alfabéticos
        """
        words = text.split()
        if keep_numbers:
            # Mantener palabras alfabéticas y números
            filtered_words = [
                word for word in words 
                if word.isalpha() or word.isdigit()
            ]
        else:
            # Solo mantener palabras alfabéticas
            filtered_words = [
                word for word in words 
                if word.isalpha()
            ]
        
        return ' '.join(filtered_words)
    
    def remove_by_pos(self, text: str, pos_to_remove: List[str] = None) -> str:
        """
        Elimina palabras según su categoría gramatical
        
        Args:
            text: Texto a procesar
            pos_to_remove: Lista de POS tags a eliminar
        """
        if pos_to_remove is None:
            # Por defecto, eliminar determinantes, preposiciones, conjunciones
            pos_to_remove = ['DET', 'ADP', 'CONJ', 'CCONJ', 'SCONJ']
        
        doc = nlp(text)
        filtered_words = [
            token.text for token in doc 
            if token.pos_ not in pos_to_remove and not token.is_space
        ]
        
        return ' '.join(filtered_words)
    
    def remove_environmental_noise(self, text: str) -> str:
        """
        Elimina ruido específico del dominio ambiental
        """
        # Palabras muy generales en contexto ambiental que no aportan información específica
        env_noise = {
            'importante', 'necesario', 'fundamental', 'esencial', 'básico',
            'general', 'específico', 'particular', 'especial', 'normal',
            'actual', 'presente', 'futuro', 'pasado', 'nuevo', 'viejo',
            'grande', 'pequeño', 'mayor', 'menor', 'mejor', 'peor',
            'bueno', 'malo', 'positivo', 'negativo', 'principal', 'secundario'
        }
        
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower() not in env_noise
        ]
        
        return ' '.join(filtered_words)
    
    def advanced_noise_removal(self, text: str) -> Dict:
        """
        Eliminación avanzada de ruido usando spaCy
        """
        doc = nlp(text)
        
        # Criterios para mantener tokens
        kept_tokens = []
        removed_tokens = []
        
        for token in doc:
            # Criterios para eliminar
            should_remove = (
                token.is_stop or           # Es stopword
                token.is_punct or          # Es puntuación
                token.is_space or          # Es espacio
                len(token.text) < 3 or     # Muy corto
                not token.is_alpha or      # No es alfabético
                token.pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ'] or  # POS irrelevantes
                token.text.lower() in self.noise_words  # Palabras de ruido
            )
            
            if should_remove:
                removed_tokens.append({
                    'text': token.text,
                    'reason': self._get_removal_reason(token)
                })
            else:
                kept_tokens.append(token.lemma_.lower())
        
        return {
            'cleaned_text': ' '.join(kept_tokens),
            'original_tokens': len(doc),
            'kept_tokens': len(kept_tokens),
            'removed_tokens': len(removed_tokens),
            'removal_details': removed_tokens[:20]  # Mostrar solo los primeros 20
        }
    
    def _get_removal_reason(self, token) -> str:
        """Determina la razón por la cual se elimina un token"""
        if token.is_stop:
            return 'stopword'
        elif token.is_punct:
            return 'punctuation'
        elif token.is_space:
            return 'whitespace'
        elif len(token.text) < 3:
            return 'too_short'
        elif not token.is_alpha:
            return 'non_alphabetic'
        elif token.pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ']:
            return 'irrelevant_pos'
        elif token.text.lower() in self.noise_words:
            return 'noise_word'
        else:
            return 'other'
    
    def comprehensive_noise_removal(self, text: str, options: Dict = None) -> Dict:
        """
        Eliminación completa de ruido con múltiples estrategias
        
        Args:
            text: Texto a limpiar
            options: Opciones de limpieza
        
        Returns:
            Diccionario con texto limpio y estadísticas
        """
        if options is None:
            options = {
                'remove_stopwords': True,
                'remove_short_words': True,
                'min_word_length': 3,
                'remove_noise_words': True,
                'remove_by_frequency': True,
                'min_frequency': 2,
                'max_frequency_ratio': 0.15,
                'remove_non_alphabetic': True,
                'keep_numbers': False,
                'remove_by_pos': True,
                'remove_environmental_noise': True,
                'use_advanced_removal': True
            }
        
        original_text = text
        processing_steps = []
        
        # Aplicar eliminaciones según opciones
        if options.get('remove_stopwords', True):
            text = self.remove_stopwords(text)
            processing_steps.append('stopwords')
        
        if options.get('remove_noise_words', True):
            text = self.remove_noise_words(text)
            processing_steps.append('noise_words')
        
        if options.get('remove_short_words', True):
            min_length = options.get('min_word_length', 3)
            text = self.remove_short_words(text, min_length)
            processing_steps.append('short_words')
        
        if options.get('remove_environmental_noise', True):
            text = self.remove_environmental_noise(text)
            processing_steps.append('environmental_noise')
        
        if options.get('remove_non_alphabetic', True):
            keep_nums = options.get('keep_numbers', False)
            text = self.remove_non_alphabetic(text, keep_nums)
            processing_steps.append('non_alphabetic')
        
        if options.get('remove_by_pos', True):
            text = self.remove_by_pos(text)
            processing_steps.append('pos_filtering')
        
        # Eliminación por frecuencia (al final para tener estadísticas correctas)
        freq_result = None
        if options.get('remove_by_frequency', True):
            min_freq = options.get('min_frequency', 2)
            max_ratio = options.get('max_frequency_ratio', 0.15)
            freq_result = self.remove_by_frequency(text, min_freq, max_ratio)
            text = freq_result['filtered_text']
            processing_steps.append('frequency_filtering')
        
        # Eliminación avanzada (opcional)
        advanced_result = None
        if options.get('use_advanced_removal', True):
            advanced_result = self.advanced_noise_removal(text)
            text = advanced_result['cleaned_text']
            processing_steps.append('advanced_removal')
        
        # Calcular estadísticas finales
        original_words = len(original_text.split())
        final_words = len(text.split())
        reduction_percentage = ((original_words - final_words) / original_words) * 100 if original_words > 0 else 0
        
        result = {
            'original_text': original_text,
            'cleaned_text': text,
            'original_word_count': original_words,
            'final_word_count': final_words,
            'words_removed': original_words - final_words,
            'reduction_percentage': round(reduction_percentage, 2),
            'processing_steps': processing_steps,
            'options_used': options
        }
        
        # Agregar resultados específicos si están disponibles
        if freq_result:
            result['frequency_analysis'] = freq_result
        if advanced_result:
            result['advanced_analysis'] = advanced_result
        
        return result