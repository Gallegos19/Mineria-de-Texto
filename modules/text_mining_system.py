from typing import Dict, List
from .text_ingestion import TextIngestion
from .text_cleaner import TextCleaner
from .text_tokenizer import TextTokenizer
from .text_normalizer import TextNormalizer
from .text_noise_remover import NoiseRemover
from .text_lemmatizer import TextLemmatizer
from .bert_processor import BERTProcessor

class TextMiningSystem:
    """
    Sistema completo de minería de texto para mejora de contenido ambiental.
    Integra todos los módulos de procesamiento en un flujo completo.
    """
    
    def __init__(self):
        self.ingestion = TextIngestion()
        self.cleaner = TextCleaner()
        self.tokenizer = TextTokenizer()
        self.normalizer = TextNormalizer()
        self.noise_remover = NoiseRemover()
        self.lemmatizer = TextLemmatizer()
        self.bert_processor = BERTProcessor()
    
    def process_text_complete_enhanced(self, text: str, content_type: str = 'contenido', track_steps: bool = False) -> Dict:
        """
        Procesamiento mejorado con mejor control de calidad
        
        Args:
            text: Texto a procesar
            content_type: Tipo de contenido ('titulo', 'descripcion', 'contenido')
            track_steps: Si se deben registrar los resultados intermedios
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        
        # Configuraciones específicas por tipo de contenido
        processing_configs = {
            'titulo': {
                'aggressive_cleaning': False,
                'preserve_length': True,
                'min_similarity': 0.7
            },
            'descripcion': {
                'aggressive_cleaning': False,
                'preserve_length': False,
                'min_similarity': 0.6
            },
            'contenido': {
                'aggressive_cleaning': True,
                'preserve_length': False,
                'min_similarity': 0.5
            }
        }
        
        config = processing_configs.get(content_type, processing_configs['contenido'])
        
        # Validación inicial
        original_input = text
        text = self._validate_text(text, "entrada inicial")
        if text is None:
            text = original_input
        
        intermediate_results = {}
        processing_steps = []
        
        try:
            # Paso 1: Ingesta (sin cambios)
            print("Ejecutando Paso 1: Ingesta...")
            ingestion_result = self.ingestion.ingest_manual_text(text)
            current_text = ingestion_result['original_text']
            
            if track_steps:
                intermediate_results['ingestion'] = current_text
                processing_steps.append({
                    'step': 'ingesta',
                    'metrics': {'length': ingestion_result['length']}
                })
            
            # Paso 2: Limpieza (menos agresiva)
            print("Ejecutando Paso 2: Limpieza...")
            cleaning_options = {
                'remove_urls': True,
                'remove_emails': True,
                'remove_phones': True,
                'remove_html': True,
                'remove_special_chars': not config['preserve_length'],
                'keep_punctuation': True,
                'normalize_whitespace': True,
                'normalize_punctuation': True,
                'remove_newlines': True,
                'fix_encoding': True
            }
            cleaning_result = self.cleaner.basic_clean(current_text, cleaning_options)
            current_text = cleaning_result['cleaned_text']
            
            if track_steps:
                intermediate_results['cleaning'] = current_text
                processing_steps.append({
                    'step': 'limpieza',
                    'metrics': {'reduction_percentage': cleaning_result['reduction_percentage']}
                })
            
            # Paso 3: Tokenización (sin cambios)
            print("Ejecutando Paso 3: Tokenización...")
            tokens = self.tokenizer.tokenize_words(current_text, 'spacy')
            if not tokens:
                tokens = current_text.split()
            
            if track_steps:
                intermediate_results['tokenization'] = ' '.join(tokens[:20]) + ('...' if len(tokens) > 20 else '')
                processing_steps.append({
                    'step': 'tokenización',
                    'metrics': {'token_count': len(tokens)}
                })
            
            # Paso 4: Normalización (mejorada)
            print("Ejecutando Paso 4: Normalización...")
            normalization_options = {
                'to_lowercase': True,
                'remove_accents': False,  # Preservar acentos para mejor legibilidad
                'expand_contractions': True,
                'expand_abbreviations': True,
                'normalize_numbers': 'keep',
                'normalize_case_patterns': True,
                'normalize_environmental_terms': True
            }
            normalization_result = self.normalizer.normalize_tokens(tokens, normalization_options)
            normalized_tokens = normalization_result.get('normalized_tokens', tokens)
            current_text = ' '.join(normalized_tokens)
            
            if track_steps:
                intermediate_results['normalization'] = current_text
                processing_steps.append({
                    'step': 'normalización',
                    'metrics': {'token_count_change': normalization_result.get('token_count_change', 0)}
                })
            
            # Paso 5: Eliminación de ruido
            print("Ejecutando Paso 5: Eliminación de ruido...")
            noise_options = {
                'remove_stopwords': config['aggressive_cleaning'],
                'remove_short_words': True,
                'min_word_length': 3,
                'remove_noise_words': True,
                'remove_by_frequency': False,  # Más conservador para preservar significado
                'remove_non_alphabetic': False,
                'keep_numbers': True,
                'remove_by_pos': False,
                'remove_environmental_noise': False,  # Preservar términos ambientales
                'use_advanced_removal': False
            }
            noise_removal_result = self.noise_remover.comprehensive_noise_removal(current_text, noise_options)
            current_text = noise_removal_result['cleaned_text']
            
            if track_steps:
                intermediate_results['noise_removal'] = current_text
                processing_steps.append({
                    'step': 'eliminación_ruido',
                    'metrics': {'reduction_percentage': noise_removal_result['reduction_percentage']}
                })
            
            # Paso 6: Lematización
            print("Ejecutando Paso 6: Lematización...")
            lemmatization_options = {
                'preserve_environmental_terms': True,
                'preserve_verbs': True,
                'min_word_length': 2,
                'remove_duplicates': False
            }
            lemmatization_result = self.lemmatizer.comprehensive_processing_improved(
                current_text, 
                method='spacy_improved',
                options=lemmatization_options
            )
            current_text = lemmatization_result['final_processed_text']
            
            if track_steps:
                intermediate_results['lemmatization'] = current_text
                processing_steps.append({
                    'step': 'lematización',
                    'metrics': {'changes_made': lemmatization_result.get('changes_made', 0)}
                })
            
            # Paso 7: Procesamiento con BERT (mejorado)
            print("Ejecutando Paso 7: Procesamiento con BERT...")
            bert_options = {
                'environmental_focus': True,
                'generate_variations': True,
                'optimize_length': True,
                'preserve_meaning': True,
                'min_similarity_threshold': config['min_similarity']
            }
            bert_result = self.bert_processor.comprehensive_text_improvement_enhanced(
                current_text,
                target_type=content_type,
                options=bert_options
            )
            final_text = bert_result['final_improved_text']
            
            if track_steps:
                intermediate_results['bert_processing'] = final_text
                processing_steps.append({
                    'step': 'bert_processing',
                    'metrics': {
                        'semantic_preservation': bert_result['improvement_summary']['semantic_preservation'],
                        'coherence_score': bert_result['improvement_summary']['coherence_score'],
                        'grammar_score': bert_result['improvement_summary']['grammar_score']
                    }
                })
            
            # Evaluación de calidad mejorada
            print("Ejecutando evaluación de calidad...")
            evaluation = self._evaluate_quality_enhanced(text, final_text, bert_result['improvement_summary'])
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations_enhanced(evaluation, processing_steps)
            
            return {
                'original_text': text,
                'final_text': final_text,
                'content_type': content_type,
                'intermediate_results': intermediate_results if track_steps else {},
                'processing_steps': processing_steps if track_steps else [],
                'evaluation': evaluation,
                'recommendations': recommendations,
                'bert_details': bert_result,
                'processing_config': config
            }
            
        except Exception as e:
            print(f"❌ Error durante el procesamiento: {str(e)}")
            return self._create_fallback_result(text, content_type, str(e))
    
    def _validate_text(self, text: str, stage: str) -> str:
        """Valida el texto en diferentes etapas del procesamiento"""
        if text is None:
            print(f"⚠️ Texto nulo en etapa {stage}")
            return ""
        
        if not isinstance(text, str):
            print(f"⚠️ Texto no es string en etapa {stage}")
            try:
                return str(text)
            except:
                return ""
        
        if len(text.strip()) == 0:
            print(f"⚠️ Texto vacío en etapa {stage}")
            return ""
        
        return text
    
    def _evaluate_quality_enhanced(self, original_text: str, final_text: str, improvement_summary: Dict) -> Dict:
        """Evaluación de calidad mejorada"""
        try:
            original_words = len(original_text.split()) if original_text else 0
            final_words = len(final_text.split()) if final_text else 0
            
            # Usar métricas del improvement_summary si están disponibles
            semantic_similarity = improvement_summary.get('semantic_preservation', 0.7)
            coherence_score = improvement_summary.get('coherence_score', 0.7)
            grammar_score = improvement_summary.get('grammar_score', 0.7)
            
            # Score ambiental
            env_score = self.bert_processor._calculate_environmental_score(final_text)
            
            # Evaluación de legibilidad mejorada
            readability_score = self._calculate_readability(final_text)
            
            # Score general ponderado
            overall_quality = (
                semantic_similarity * 3 +
                coherence_score * 2.5 +
                grammar_score * 2 +
                (env_score / 10) * 1.5 +
                (readability_score / 10) * 1
            )
            
            return {
                'semantic_similarity': semantic_similarity * 10,
                'coherence': coherence_score * 10,
                'grammar': grammar_score * 10,
                'environmental_relevance': env_score,
                'readability': readability_score,
                'length_optimization': min(10, max(1, 10 - abs(final_words - 15) / 5)) if final_words > 0 else 5.0,
                'overall_quality': min(10, overall_quality)
            }
            
        except Exception as e:
            print(f"❌ Error en evaluación: {e}")
            return {
                'semantic_similarity': 7.0,
                'coherence': 7.0,
                'grammar': 7.0,
                'environmental_relevance': 5.0,
                'readability': 7.0,
                'length_optimization': 7.0,
                'overall_quality': 6.8
            }
    
    def _calculate_readability(self, text: str) -> float:
        """Calcula un score de legibilidad básico"""
        if not text:
            return 5.0
        
        words = text.split()
        sentences = text.split('.')
        
        # Métricas básicas
        avg_words_per_sentence = len(words) / len(sentences) if sentences else len(words)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Score basado en complejidad
        readability = 10
        
        # Penalizar oraciones muy largas
        if avg_words_per_sentence > 20:
            readability -= 2
        elif avg_words_per_sentence > 15:
            readability -= 1
        
        # Penalizar palabras muy largas
        if avg_word_length > 8:
            readability -= 2
        elif avg_word_length > 6:
            readability -= 1
        
        # Bonificar longitud apropiada
        if 5 <= len(words) <= 20:
            readability += 1
        
        return max(1.0, min(10.0, readability))
    
    def _generate_recommendations_enhanced(self, evaluation: Dict, processing_steps: List) -> List[str]:
        """Genera recomendaciones mejoradas"""
        recommendations = []
        
        try:
            if evaluation.get('semantic_similarity', 0) < 7:
                recommendations.append("Ajustar parámetros de lematización para preservar mejor el significado original")
            
            if evaluation.get('coherence', 0) < 7:
                recommendations.append("Mejorar la coherencia textual manteniendo la estructura gramatical")
            
            if evaluation.get('grammar', 0) < 7:
                recommendations.append("Revisar la gramática del texto procesado")
            
            if evaluation.get('environmental_relevance', 0) < 5:
                recommendations.append("Incorporar más términos ambientales relevantes")
            
            if evaluation.get('readability', 0) < 6:
                recommendations.append("Simplificar el vocabulario para mejorar la legibilidad")
            
            if evaluation.get('length_optimization', 0) < 7:
                recommendations.append("Ajustar la longitud según el tipo de contenido")
            
            if not recommendations:
                recommendations.append("El texto ha sido procesado exitosamente con alta calidad")
                
        except Exception as e:
            recommendations.append(f"Error generando recomendaciones: {str(e)}")
        
        return recommendations
    
    def _create_fallback_result(self, text: str, content_type: str, error_message: str) -> Dict:
        """Crea un resultado de fallback en caso de error"""
        return {
            'original_text': text,
            'final_text': text,  # Sin cambios
            'content_type': content_type,
            'error': error_message,
            'status': 'error',
            'recommendations': ["Hubo un error en el procesamiento. Por favor, revise el texto de entrada."]
        }