import re
import numpy as np
from typing import Dict, List
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize

# Load spaCy model
try:
    nlp = spacy.load('es_core_news_sm')
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load('es_core_news_sm')

class TextTokenizer:
    """
    Módulo para la tokenización del texto.
    Divide el texto en tokens (palabras, oraciones, párrafos) usando diferentes estrategias.
    """
    
    def __init__(self):
        # Configurar tokenizadores
        self.nlp = nlp  # spaCy model cargado anteriormente
        
        # Patrones para tokenización personalizada
        self.sentence_endings = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        
    def tokenize_sentences(self, text: str, method: str = 'nltk') -> List[str]:
        """
        Tokeniza el texto en oraciones
        
        Args:
            text: Texto a tokenizar
            method: Método a usar ('nltk', 'spacy', 'regex')
        
        Returns:
            Lista de oraciones
        """
        if method == 'nltk':
            sentences = sent_tokenize(text, language='spanish')
        elif method == 'spacy':
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        elif method == 'regex':
            # Método simple con regex
            sentences = self.sentence_endings.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            raise ValueError("Método debe ser 'nltk', 'spacy' o 'regex'")
        
        return [s for s in sentences if len(s.strip()) > 0]
    
    def tokenize_words(self, text: str, method: str = 'nltk') -> List[str]:
        """
        Tokeniza el texto en palabras
        
        Args:
            text: Texto a tokenizar
            method: Método a usar ('nltk', 'spacy', 'regex')
        
        Returns:
            Lista de palabras
        """
        if method == 'nltk':
            words = word_tokenize(text, language='spanish')
        elif method == 'spacy':
            doc = self.nlp(text)
            words = [token.text for token in doc if not token.is_space]
        elif method == 'regex':
            words = self.word_pattern.findall(text)
        else:
            raise ValueError("Método debe ser 'nltk', 'spacy' o 'regex'")
        
        return words
    
    def tokenize_paragraphs(self, text: str) -> List[str]:
        """
        Tokeniza el texto en párrafos
        """
        # Dividir por dobles saltos de línea o líneas vacías
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def advanced_tokenization(self, text: str) -> Dict:
        """
        Tokenización avanzada usando spaCy con información lingüística
        
        Returns:
            Diccionario con tokens y sus propiedades
        """
        doc = self.nlp(text)
        
        tokens_info = []
        for token in doc:
            if not token.is_space:
                token_info = {
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,  # Part of speech
                    'tag': token.tag_,  # Detailed POS tag
                    'is_alpha': token.is_alpha,
                    'is_stop': token.is_stop,
                    'is_punct': token.is_punct,
                    'is_digit': token.is_digit,
                    'shape': token.shape_,  # Forma de la palabra (Xxxx, dddd, etc.)
                    'is_title': token.is_title,
                    'is_lower': token.is_lower,
                    'is_upper': token.is_upper
                }
                tokens_info.append(token_info)
        
        return {
            'tokens': tokens_info,
            'total_tokens': len(tokens_info),
            'sentences': [sent.text for sent in doc.sents],
            'entities': [(ent.text, ent.label_) for ent in doc.ents]
        }
    
    def tokenize_by_pos(self, text: str) -> Dict[str, List[str]]:
        """
        Tokeniza y agrupa por categorías gramaticales
        
        Returns:
            Diccionario con tokens agrupados por POS
        """
        doc = self.nlp(text)
        
        pos_groups = {
            'sustantivos': [],
            'verbos': [],
            'adjetivos': [],
            'adverbios': [],
            'preposiciones': [],
            'conjunciones': [],
            'determinantes': [],
            'pronombres': [],
            'otros': []
        }
        
        pos_mapping = {
            'NOUN': 'sustantivos',
            'VERB': 'verbos',
            'ADJ': 'adjetivos',
            'ADV': 'adverbios',
            'ADP': 'preposiciones',
            'CONJ': 'conjunciones',
            'CCONJ': 'conjunciones',
            'DET': 'determinantes',
            'PRON': 'pronombres'
        }
        
        for token in doc:
            if token.is_alpha and not token.is_stop:
                category = pos_mapping.get(token.pos_, 'otros')
                pos_groups[category].append(token.lemma_.lower())
        
        # Eliminar duplicados manteniendo orden
        for category in pos_groups:
            pos_groups[category] = list(dict.fromkeys(pos_groups[category]))
        
        return pos_groups
    
    def extract_environmental_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Extrae términos específicos relacionados con medio ambiente
        """
        doc = self.nlp(text)
        
        # Términos ambientales por categoría
        environmental_categories = {
            'problemas_ambientales': [],
            'soluciones': [],
            'recursos_naturales': [],
            'energia': [],
            'contaminacion': [],
            'conservacion': []
        }
        
        # Diccionarios de términos ambientales
        environmental_terms = {
            'problemas_ambientales': [
                'cambio climático', 'calentamiento global', 'deforestación', 
                'contaminación', 'extinción', 'desertificación', 'erosión'
            ],
            'soluciones': [
                'reciclaje', 'energía renovable', 'sostenible', 'sustentable',
                'conservación', 'reforestación', 'eficiencia energética'
            ],
            'recursos_naturales': [
                'agua', 'bosque', 'océano', 'biodiversidad', 'ecosistema',
                'fauna', 'flora', 'selva', 'río', 'lago'
            ],
            'energia': [
                'solar', 'eólica', 'hidroeléctrica', 'geotérmica', 
                'biomasa', 'combustible fósil', 'petróleo', 'carbón'
            ],
            'contaminacion': [
                'emisiones', 'gases', 'residuos', 'basura', 'tóxicos',
                'plástico', 'químicos', 'desechos'
            ],
            'conservacion': [
                'protección', 'preservación', 'área protegida', 'parque nacional',
                'reserva', 'santuario', 'hábitat'
            ]
        }
        
        text_lower = text.lower()
        
        for category, terms in environmental_terms.items():
            for term in terms:
                if term in text_lower:
                    environmental_categories[category].append(term)
        
        return environmental_categories
    
    def tokenization_statistics(self, text: str) -> Dict:
        """
        Genera estadísticas completas de tokenización
        """
        sentences = self.tokenize_sentences(text, 'spacy')
        words = self.tokenize_words(text, 'spacy')
        paragraphs = self.tokenize_paragraphs(text)
        advanced_tokens = self.advanced_tokenization(text)
        pos_groups = self.tokenize_by_pos(text)
        env_terms = self.extract_environmental_terms(text)
        
        # Calcular estadísticas
        word_lengths = [len(word) for word in words if word.isalpha()]
        sentence_lengths = [len(sent.split()) for sent in sentences]
        
        stats = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'total_paragraphs': len(paragraphs),
            'unique_words': len(set(word.lower() for word in words if word.isalpha())),
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'pos_distribution': {k: len(v) for k, v in pos_groups.items()},
            'environmental_terms_count': sum(len(terms) for terms in env_terms.values()),
            'entities_found': len(advanced_tokens['entities']),
            'lexical_diversity': len(set(word.lower() for word in words if word.isalpha())) / len(words) if words else 0
        }
        
        return stats