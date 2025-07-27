import re
from typing import Dict, List, Set
import spacy
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('es_core_news_sm')
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load('es_core_news_sm')

class TextLemmatizer:
    """
    Módulo para lematización y stemming del texto.
    Reduce las palabras a su forma base usando diferentes estrategias.
    Incluye evaluación comparativa entre métodos.
    """
    
    def __init__(self):
        # Configurar herramientas
        self.nlp = nlp  # spaCy model
        self.stemmer = SnowballStemmer('spanish')
        self.nltk_lemmatizer = WordNetLemmatizer()  # Para comparación
        
        # Diccionario de lemas personalizados para términos ambientales
        self.environmental_lemmas = {
            'contaminaciones': 'contaminación',
            'deforestaciones': 'deforestación',
            'reforestaciones': 'reforestación',
            'sostenibilidades': 'sostenibilidad',
            'biodiversidades': 'biodiversidad',
            'ecosistemas': 'ecosistema',
            'energías': 'energía',
            'renovables': 'renovable',
            'combustibles': 'combustible',
            'emisiones': 'emisión',
            'residuos': 'residuo',
            'desechos': 'desecho',
            'conservaciones': 'conservación',
            'preservaciones': 'preservación',
            'protecciones': 'protección',
            'calentamientos': 'calentamiento',
            'cambios': 'cambio',
            'efectos': 'efecto',
            'impactos': 'impacto',
            'consecuencias': 'consecuencia',
            'soluciones': 'solución',
            'alternativas': 'alternativa',
            'tecnologías': 'tecnología',
            'innovaciones': 'innovación'
        }
        
        # Diccionario de verbos en español con formas irregulares
        self.spanish_verbs = {
            # Verbos comunes con conjugaciones problemáticas
            'cuidemos': 'cuidar',
            'cuider': 'cuidar',
            'protejamos': 'proteger',
            'conservemos': 'conservar',
            'reciclemos': 'reciclar',
            'reduzcamos': 'reducir',
            'reutilicemos': 'reutilizar',
            'ahorremos': 'ahorrar',
            'plantemos': 'plantar',
            'sembremos': 'sembrar',
            'limpiemos': 'limpiar',
            'preservemos': 'preservar',
            'defendamos': 'defender',
            'salvemos': 'salvar',
            'ayudemos': 'ayudar',
            'mejoremos': 'mejorar',
            'transformemos': 'transformar',
            'cambiemos': 'cambiar',
            'actuemos': 'actuar',
            'pensemos': 'pensar'
        }
        
        # Excepciones de stemming (palabras que no deben ser stemmed)
        self.stemming_exceptions = {
            'gases', 'atlas', 'crisis', 'análisis', 'síntesis',
            'tesis', 'oasis', 'énfasis', 'paréntesis'
        }
    
    def lemmatize_with_spacy(self, text: str) -> Dict:
        """
        Lematización usando spaCy (método recomendado)
        
        Returns:
            Diccionario con texto lematizado y análisis detallado
        """
        doc = self.nlp(text)
        
        lemmatized_tokens = []
        lemmatization_details = []
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                original = token.text
                lemma = token.lemma_.lower()
                
                # Aplicar lemas personalizados para términos ambientales
                if original.lower() in self.environmental_lemmas:
                    lemma = self.environmental_lemmas[original.lower()]
                
                # Aplicar lemas personalizados para verbos españoles
                elif original.lower() in self.spanish_verbs:
                    lemma = self.spanish_verbs[original.lower()]
                
                # Verificar si el lemma generado está en nuestro diccionario de verbos
                elif lemma in self.spanish_verbs:
                    lemma = self.spanish_verbs[lemma]
                
                lemmatized_tokens.append(lemma)
                
                # Guardar detalles si hay cambio
                if original.lower() != lemma:
                    lemmatization_details.append({
                        'original': original,
                        'lemma': lemma,
                        'pos': token.pos_,
                        'change_type': self._get_change_type(original, lemma)
                    })
        
        return {
            'lemmatized_text': ' '.join(lemmatized_tokens),
            'original_tokens': len([t for t in doc if not t.is_space and not t.is_punct]),
            'lemmatized_tokens': len(lemmatized_tokens),
            'changes_made': len(lemmatization_details),
            'lemmatization_details': lemmatization_details,
            'method': 'spacy'
        }
    
    def stem_with_snowball(self, text: str) -> Dict:
        """
        Stemming usando SnowballStemmer
        
        Returns:
            Diccionario con texto stemmed y análisis detallado
        """
        words = text.split()
        
        stemmed_tokens = []
        stemming_details = []
        
        for word in words:
            if word.isalpha():
                original = word.lower()
                
                # Verificar excepciones
                if original in self.stemming_exceptions:
                    stemmed = original
                else:
                    stemmed = self.stemmer.stem(original)
                
                stemmed_tokens.append(stemmed)
                
                # Guardar detalles si hay cambio
                if original != stemmed:
                    stemming_details.append({
                        'original': word,
                        'stem': stemmed,
                        'change_type': self._get_change_type(original, stemmed)
                    })
            else:
                stemmed_tokens.append(word)
        
        return {
            'stemmed_text': ' '.join(stemmed_tokens),
            'original_tokens': len(words),
            'stemmed_tokens': len(stemmed_tokens),
            'changes_made': len(stemming_details),
            'stemming_details': stemming_details,
            'method': 'snowball'
        }
    
    def hybrid_approach(self, text: str) -> Dict:
        """
        Enfoque híbrido: lematización para sustantivos/adjetivos, stemming para verbos
        """
        doc = self.nlp(text)
        
        processed_tokens = []
        processing_details = []
        
        for token in doc:
            if not token.is_space and not token.is_punct and token.is_alpha:
                original = token.text
                
                # Decidir método según POS
                if token.pos_ in ['NOUN', 'ADJ', 'PROPN']:
                    # Usar lematización para sustantivos y adjetivos
                    processed = token.lemma_.lower()
                    method_used = 'lemmatization'
                    
                    # Aplicar lemas personalizados
                    if original.lower() in self.environmental_lemmas:
                        processed = self.environmental_lemmas[original.lower()]
                        method_used = 'custom_lemma'
                        
                elif token.pos_ in ['VERB', 'AUX']:
                    # Usar stemming para verbos
                    if original.lower() not in self.stemming_exceptions:
                        processed = self.stemmer.stem(original.lower())
                        method_used = 'stemming'
                    else:
                        processed = original.lower()
                        method_used = 'exception'
                else:
                    # Para otras categorías, usar lematización
                    processed = token.lemma_.lower()
                    method_used = 'lemmatization'
                
                processed_tokens.append(processed)
                
                # Guardar detalles si hay cambio
                if original.lower() != processed:
                    processing_details.append({
                        'original': original,
                        'processed': processed,
                        'pos': token.pos_,
                        'method': method_used,
                        'change_type': self._get_change_type(original, processed)
                    })
        
        return {
            'processed_text': ' '.join(processed_tokens),
            'original_tokens': len([t for t in doc if not t.is_space and not t.is_punct and t.is_alpha]),
            'processed_tokens': len(processed_tokens),
            'changes_made': len(processing_details),
            'processing_details': processing_details,
            'method': 'hybrid'
        }
    
    def _get_change_type(self, original: str, processed: str) -> str:
        """Determina el tipo de cambio realizado"""
        if len(processed) < len(original):
            return 'reduction'
        elif len(processed) > len(original):
            return 'expansion'
        elif processed != original.lower():
            return 'transformation'
        else:
            return 'no_change'
    
    def compare_methods(self, text: str) -> Dict:
        """
        Compara los tres métodos de procesamiento
        """
        # Aplicar cada método
        spacy_result = self.lemmatize_with_spacy(text)
        snowball_result = self.stem_with_snowball(text)
        hybrid_result = self.hybrid_approach(text)
        
        # Calcular métricas de comparación
        original_words = set(text.lower().split())
        spacy_words = set(spacy_result['lemmatized_text'].split())
        snowball_words = set(snowball_result['stemmed_text'].split())
        hybrid_words = set(hybrid_result['processed_text'].split())
        
        # Calcular reducción de vocabulario
        vocab_reduction = {
            'original': len(original_words),
            'spacy': len(spacy_words),
            'snowball': len(snowball_words),
            'hybrid': len(hybrid_words)
        }
        
        # Calcular porcentajes de reducción
        reduction_percentages = {}
        for method, vocab_size in vocab_reduction.items():
            if method != 'original':
                reduction_percentages[method] = round(
                    ((vocab_reduction['original'] - vocab_size) / vocab_reduction['original']) * 100, 2
                ) if vocab_reduction['original'] > 0 else 0
        
        return {
            'results': {
                'spacy': spacy_result,
                'snowball': snowball_result,
                'hybrid': hybrid_result
            },
            'vocabulary_sizes': vocab_reduction,
            'reduction_percentages': reduction_percentages,
            'comparison_summary': {
                'most_aggressive': min(vocab_reduction, key=vocab_reduction.get),
                'most_conservative': max(vocab_reduction, key=vocab_reduction.get),
                'recommended': 'spacy'  # Nuestra recomendación inicial
            }
        }
    
    def comprehensive_processing(self, text: str, method: str = 'spacy', options: Dict = None) -> Dict:
        """
        Procesamiento completo con el método seleccionado
        
        Args:
            text: Texto a procesar
            method: 'spacy', 'snowball', 'hybrid', o 'compare'
            options: Opciones adicionales
        """
        if options is None:
            options = {
                'preserve_environmental_terms': True,
                'min_word_length': 2,
                'remove_duplicates': True
            }
        
        if method == 'compare':
            return self.compare_methods(text)
        elif method == 'spacy':
            result = self.lemmatize_with_spacy(text)
        elif method == 'snowball':
            result = self.stem_with_snowball(text)
        elif method == 'hybrid':
            result = self.hybrid_approach(text)
        else:
            raise ValueError("Método debe ser 'spacy', 'snowball', 'hybrid', o 'compare'")
        
        # Aplicar opciones adicionales
        processed_text = result.get('lemmatized_text') or result.get('stemmed_text') or result.get('processed_text')
        
        if options.get('min_word_length', 2) > 1:
            words = processed_text.split()
            words = [w for w in words if len(w) >= options['min_word_length']]
            processed_text = ' '.join(words)
        
        if options.get('remove_duplicates', True):
            words = processed_text.split()
            # Mantener orden pero eliminar duplicados
            seen = set()
            unique_words = []
            for word in words:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            processed_text = ' '.join(unique_words)
        
        # Actualizar resultado
        result['final_processed_text'] = processed_text
        result['options_applied'] = options
        
        return result
    
    def comprehensive_processing_improved(self, text: str, method: str = 'spacy_improved', options: Dict = None) -> Dict:
        """
        Versión mejorada del procesamiento completo
        
        Args:
            text: Texto a procesar
            method: Método de procesamiento
            options: Opciones adicionales
        """
        # Procesar con el método base
        result = self.comprehensive_processing(text, 'spacy' if method == 'spacy_improved' else method, options)
        
        # Añadir mejoras adicionales
        if method == 'spacy_improved':
            # Preservar términos ambientales importantes
            if options and options.get('preserve_verbs', False):
                # Identificar verbos importantes y restaurarlos
                doc = self.nlp(text)
                important_verbs = [token.text.lower() for token in doc if token.pos_ == 'VERB' and len(token.text) > 3]
                
                # Añadir información sobre verbos preservados
                result['preserved_verbs'] = important_verbs
        
        return result