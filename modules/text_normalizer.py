import re
import unicodedata
from typing import Dict, List

class TextNormalizer:
    """
    Módulo para la normalización del texto.
    Convierte texto a formas estándar: minúsculas, eliminación de acentos, 
    expansión de contracciones, normalización de números, etc.
    """
    
    def __init__(self):
        # Diccionario de contracciones en español
        self.contractions = {
            'del': 'de el',
            'al': 'a el',
            'pa': 'para',
            'pal': 'para el',
            'pá': 'para',
            'q': 'que',
            'xq': 'porque',
            'porq': 'porque',
            'x': 'por',
            'tb': 'también',
            'tmb': 'también',
            'tbn': 'también',
            'pq': 'porque',
            'xk': 'porque',
            'k': 'que',
            'dnd': 'donde',
            'dónde': 'donde',
            'cuándo': 'cuando',
            'cómo': 'como',
            'qué': 'que',
            'cuál': 'cual',
            'cuáles': 'cuales',
            'quién': 'quien',
            'quiénes': 'quienes'
        }
        
        # Abreviaciones comunes
        self.abbreviations = {
            'dr.': 'doctor',
            'dra.': 'doctora',
            'sr.': 'señor',
            'sra.': 'señora',
            'srta.': 'señorita',
            'prof.': 'profesor',
            'profa.': 'profesora',
            'ing.': 'ingeniero',
            'lic.': 'licenciado',
            'etc.': 'etcétera',
            'vs.': 'versus',
            'ej.': 'ejemplo',
            'p.ej.': 'por ejemplo',
            'i.e.': 'es decir',
            'e.g.': 'por ejemplo',
            'aprox.': 'aproximadamente',
            'máx.': 'máximo',
            'mín.': 'mínimo',
            'kg.': 'kilogramos',
            'km.': 'kilómetros',
            'm.': 'metros',
            'cm.': 'centímetros',
            'mm.': 'milímetros',
            'co2': 'dióxido de carbono',
            'ong': 'organización no gubernamental',
            'onu': 'organización de las naciones unidas'
        }
        
        # Números escritos
        self.number_words = {
            'cero': '0', 'uno': '1', 'dos': '2', 'tres': '3', 'cuatro': '4',
            'cinco': '5', 'seis': '6', 'siete': '7', 'ocho': '8', 'nueve': '9',
            'diez': '10', 'once': '11', 'doce': '12', 'trece': '13', 'catorce': '14',
            'quince': '15', 'dieciséis': '16', 'diecisiete': '17', 'dieciocho': '18',
            'diecinueve': '19', 'veinte': '20', 'treinta': '30', 'cuarenta': '40',
            'cincuenta': '50', 'sesenta': '60', 'setenta': '70', 'ochenta': '80',
            'noventa': '90', 'cien': '100', 'mil': '1000', 'millón': '1000000'
        }
        
        # Términos ambientales para normalización
        self.env_normalizations = {
            'co2': 'dióxido de carbono',
            'co₂': 'dióxido de carbono',
            'ch4': 'metano',
            'ch₄': 'metano',
            'n2o': 'óxido nitroso',
            'n₂o': 'óxido nitroso',
            'ghg': 'gases de efecto invernadero',
            'gei': 'gases de efecto invernadero',
            'renewable energy': 'energía renovable',
            'green house': 'efecto invernadero',
            'global warming': 'calentamiento global',
            'climate change': 'cambio climático',
            'sustainable development': 'desarrollo sostenible',
            'carbon footprint': 'huella de carbono',
            'biodiversity': 'biodiversidad',
            'ecosystem': 'ecosistema',
            'deforestation': 'deforestación',
            'reforestation': 'reforestación'
        }
    
    def to_lowercase(self, tokens: List[str]) -> List[str]:
        """Convierte tokens a minúsculas"""
        return [token.lower() for token in tokens]
    
    def remove_accents(self, tokens: List[str]) -> List[str]:
        """
        Elimina acentos y diacríticos de los tokens
        """
        normalized_tokens = []
        for token in tokens:
            # Normalizar usando NFD (Canonical Decomposition)
            text_nfd = unicodedata.normalize('NFD', token)
            # Filtrar caracteres diacríticos
            text_without_accents = ''.join(
                char for char in text_nfd 
                if unicodedata.category(char) != 'Mn'
            )
            normalized_tokens.append(text_without_accents)
        return normalized_tokens
    
    def expand_contractions(self, tokens: List[str]) -> List[str]:
        """
        Expande contracciones comunes en español
        """
        expanded_tokens = []
        for token in tokens:
            token_lower = token.lower()
            # Buscar contracción exacta
            if token_lower in self.contractions:
                # Dividir la expansión en tokens si contiene espacios
                expansion = self.contractions[token_lower]
                if ' ' in expansion:
                    expanded_tokens.extend(expansion.split())
                else:
                    expanded_tokens.append(expansion)
            else:
                expanded_tokens.append(token)
        return expanded_tokens
    
    def expand_abbreviations(self, tokens: List[str]) -> List[str]:
        """
        Expande abreviaciones comunes
        """
        expanded_tokens = []
        for token in tokens:
            token_lower = token.lower()
            # Verificar si el token es una abreviación
            if token_lower in self.abbreviations:
                expansion = self.abbreviations[token_lower]
                if ' ' in expansion:
                    expanded_tokens.extend(expansion.split())
                else:
                    expanded_tokens.append(expansion)
            else:
                expanded_tokens.append(token)
        return expanded_tokens
    
    def normalize_numbers(self, tokens: List[str], strategy: str = 'keep') -> List[str]:
        """
        Normaliza números en los tokens
        
        Args:
            tokens: Lista de tokens
            strategy: 'keep', 'remove', 'words_to_digits', 'digits_to_words'
        """
        if strategy == 'remove':
            # Eliminar todos los tokens que son números
            return [token for token in tokens if not token.isdigit()]
        
        elif strategy == 'words_to_digits':
            # Convertir números escritos a dígitos
            normalized_tokens = []
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.number_words:
                    normalized_tokens.append(self.number_words[token_lower])
                else:
                    normalized_tokens.append(token)
            return normalized_tokens
        
        elif strategy == 'digits_to_words':
            # Convertir dígitos simples a palabras (0-20)
            digit_to_word = {v: k for k, v in self.number_words.items() if int(v) <= 20}
            normalized_tokens = []
            for token in tokens:
                if token.isdigit() and token in digit_to_word:
                    normalized_tokens.append(digit_to_word[token])
                else:
                    normalized_tokens.append(token)
            return normalized_tokens
        
        else:  # 'keep'
            return tokens
    
    def normalize_case_patterns(self, tokens: List[str]) -> List[str]:
        """
        Normaliza patrones de mayúsculas y minúsculas
        """
        normalized_tokens = []
        for token in tokens:
            if token.isupper() and len(token) > 1:
                # Si toda la palabra está en mayúsculas, convertir a minúsculas
                normalized_tokens.append(token.lower())
            elif token.islower():
                # Si está en minúsculas, mantener
                normalized_tokens.append(token)
            else:
                # Casos mixtos, mantener como está
                normalized_tokens.append(token)
        return normalized_tokens
    
    def normalize_environmental_terms(self, tokens: List[str]) -> List[str]:
        """
        Normaliza términos específicos del dominio ambiental
        """
        # Primero, reconstruir el texto para buscar términos compuestos
        text = ' '.join(tokens).lower()
        
        # Aplicar normalizaciones
        for term, normalized in self.env_normalizations.items():
            text = re.sub(r'\b' + re.escape(term) + r'\b', normalized, text)
        
        # Volver a tokenizar
        return text.split()
    
    def normalize_tokens(self, tokens: List[str], options: Dict = None) -> Dict:
        """
        Realiza normalización completa de los tokens
        
        Args:
            tokens: Lista de tokens a normalizar
            options: Diccionario con opciones de normalización
        
        Returns:
            Diccionario con tokens normalizados y estadísticas
        """
        if options is None:
            options = {
                'to_lowercase': True,
                'remove_accents': True,
                'expand_contractions': True,
                'expand_abbreviations': True,
                'normalize_numbers': 'keep',
                'normalize_case_patterns': True,
                'normalize_environmental_terms': True
            }
        
        original_tokens = tokens.copy()
        steps_applied = []
        
        # Aplicar normalizaciones según opciones
        if options.get('normalize_case_patterns', True):
            tokens = self.normalize_case_patterns(tokens)
            steps_applied.append('case_patterns')
        
        if options.get('to_lowercase', True):
            tokens = self.to_lowercase(tokens)
            steps_applied.append('lowercase')
        
        if options.get('expand_contractions', True):
            tokens = self.expand_contractions(tokens)
            steps_applied.append('contractions')
        
        if options.get('expand_abbreviations', True):
            tokens = self.expand_abbreviations(tokens)
            steps_applied.append('abbreviations')
        
        if options.get('normalize_environmental_terms', True):
            tokens = self.normalize_environmental_terms(tokens)
            steps_applied.append('environmental_terms')
        
        if options.get('normalize_numbers', 'keep') != 'keep':
            tokens = self.normalize_numbers(tokens, options['normalize_numbers'])
            steps_applied.append('numbers')
        
        if options.get('remove_accents', True):
            tokens = self.remove_accents(tokens)
            steps_applied.append('accents')
        
        return {
            'original_tokens': original_tokens,
            'normalized_tokens': tokens,
            'original_count': len(original_tokens),
            'normalized_count': len(tokens),
            'token_count_change': len(tokens) - len(original_tokens),
            'steps_applied': steps_applied,
            'normalization_options': options
        }
    
    def comprehensive_normalization(self, text: str, options: Dict = None) -> Dict:
        """
        Realiza normalización completa del texto (para compatibilidad con versiones anteriores)
        
        Args:
            text: Texto a normalizar
            options: Diccionario con opciones de normalización
        
        Returns:
            Diccionario con texto normalizado y estadísticas
        """
        # Tokenizar el texto
        tokens = text.split()
        
        # Normalizar los tokens
        result = self.normalize_tokens(tokens, options)
        
        # Reconstruir el texto normalizado
        normalized_text = ' '.join(result['normalized_tokens'])
        
        return {
            'original_text': text,
            'normalized_text': normalized_text,
            'original_word_count': result['original_count'],
            'normalized_word_count': result['normalized_count'],
            'word_count_change': result['token_count_change'],
            'steps_applied': result['steps_applied'],
            'normalization_options': result['normalization_options']
        }
    
    def normalize_tokenized_text(self, tokenization_result: Dict, options: Dict = None) -> Dict:
        """
        Normaliza el resultado de la tokenización
        
        Args:
            tokenization_result: Resultado del módulo de tokenización
            options: Opciones de normalización
        
        Returns:
            Diccionario con resultados normalizados
        """
        # Extraer tokens del resultado de tokenización
        if 'tokens' in tokenization_result:
            # Si es resultado de advanced_tokenization
            tokens = [token['text'] for token in tokenization_result['tokens']]
        elif isinstance(tokenization_result, list):
            # Si es una lista simple de tokens
            tokens = tokenization_result
        else:
            # Intentar extraer palabras
            tokens = tokenization_result.get('words', [])
            if not tokens:
                raise ValueError("No se pudieron extraer tokens del resultado de tokenización")
        
        # Normalizar los tokens
        normalization_result = self.normalize_tokens(tokens, options)
        
        # Si el resultado original incluye información POS, mantenerla
        if 'tokens' in tokenization_result and isinstance(tokenization_result['tokens'], list):
            # Crear un mapeo de tokens originales a normalizados
            # (esto es una aproximación, ya que la normalización puede cambiar el número de tokens)
            normalized_tokens_info = []
            
            # Intentar preservar información lingüística
            for i, token_info in enumerate(tokenization_result['tokens']):
                if i < len(normalization_result['normalized_tokens']):
                    normalized_token = normalization_result['normalized_tokens'][i]
                    normalized_tokens_info.append({
                        **token_info,
                        'original_text': token_info['text'],
                        'text': normalized_token,
                        'normalized': True
                    })
            
            normalization_result['normalized_tokens_info'] = normalized_tokens_info
        
        return normalization_result