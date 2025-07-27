import numpy as np
import torch
from typing import Dict, List
from transformers import BertTokenizer, BertModel

class BERTProcessor:
    """
    Módulo para procesamiento y mejora de texto usando BERT.
    Incluye generación de embeddings, mejora de texto, y análisis semántico.
    """
    
    def __init__(self, model_name: str = 'dccuchile/bert-base-spanish-wwm-uncased'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar modelos (se cargarán cuando sea necesario)
        self.tokenizer = None
        self.model = None
        
        # Configuraciones para diferentes tareas
        self.max_length = 512
        self.environmental_keywords = [
            'medio ambiente', 'sostenible', 'ecológico', 'verde', 'limpio',
            'renovable', 'conservación', 'biodiversidad', 'clima', 'carbono',
            'emisiones', 'contaminación', 'reciclaje', 'energía', 'natural'
        ]
        
        # Templates para mejora de texto
        self.improvement_templates = {
            'titulo': "Mejora este título para que sea más atractivo y claro: {text}",
            'descripcion': "Reescribe esta descripción para que sea más informativa y engaging: {text}",
            'contenido': "Mejora este contenido haciéndolo más profesional y completo: {text}",
            'llamada_accion': "Convierte este texto en una llamada a la acción convincente: {text}",
            'resumen': "Crea un resumen conciso y impactante de: {text}"
        }
    
    def _load_bert_model(self):
        """Carga el modelo BERT si no está cargado"""
        if self.tokenizer is None or self.model is None:
            print(f"Cargando modelo BERT: {self.model_name}")
            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                print("✓ Modelo BERT cargado exitosamente")
            except Exception as e:
                print(f"Error cargando BERT: {e}")
                # Fallback a modelo en inglés
                print("Usando modelo BERT en inglés como fallback...")
                self.model_name = 'bert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
    
    def generate_embeddings(self, text: str) -> Dict:
        """
        Genera embeddings BERT para el texto
        
        Returns:
            Diccionario con embeddings y estadísticas
        """
        self._load_bert_model()
        
        # Tokenizar texto
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Mover a dispositivo
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generar embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Obtener diferentes tipos de embeddings
            last_hidden_states = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            
            # Embedding promedio de todos los tokens
            mean_embedding = torch.mean(last_hidden_states, dim=1)
            
            # Embedding del token [CLS]
            cls_embedding = last_hidden_states[:, 0, :]
        
        return {
            'text': text,
            'token_count': inputs['input_ids'].shape[1],
            'embeddings': {
                'cls_embedding': cls_embedding.cpu().numpy(),
                'mean_embedding': mean_embedding.cpu().numpy(),
                'pooler_output': pooler_output.cpu().numpy()
            },
            'embedding_dimension': last_hidden_states.shape[-1],
            'model_used': self.model_name
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> Dict:
        """
        Calcula similitud semántica entre dos textos usando BERT
        """
        # Generar embeddings para ambos textos
        emb1 = self.generate_embeddings(text1)
        emb2 = self.generate_embeddings(text2)
        
        # Calcular similitud coseno para diferentes tipos de embeddings
        similarities = {}
        
        for emb_type in ['cls_embedding', 'mean_embedding', 'pooler_output']:
            vec1 = emb1['embeddings'][emb_type].flatten()
            vec2 = emb2['embeddings'][emb_type].flatten()
            
            # Similitud coseno
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
            similarities[emb_type] = float(similarity)
        
        return {
            'text1': text1,
            'text2': text2,
            'similarities': similarities,
            'average_similarity': np.mean(list(similarities.values())),
            'most_similar_embedding': max(similarities, key=similarities.get)
        }
    
    def improve_text_with_context(self, text: str, improvement_type: str = 'contenido', 
                                 environmental_focus: bool = True) -> Dict:
        """
        Mejora el texto usando contexto ambiental y BERT
        
        Args:
            text: Texto a mejorar
            improvement_type: Tipo de mejora ('titulo', 'descripcion', 'contenido', etc.)
            environmental_focus: Si enfocar en términos ambientales
        """
        # Analizar el texto original
        original_embeddings = self.generate_embeddings(text)
        
        # Identificar términos ambientales presentes
        environmental_terms_found = [
            term for term in self.environmental_keywords 
            if term.lower() in text.lower()
        ]
        
        # Generar versiones mejoradas
        improvements = []
        
        # Mejora 1: Expansión con términos ambientales
        if environmental_focus:
            expanded_text = self._expand_with_environmental_terms(text, environmental_terms_found)
            improvements.append({
                'version': 'environmental_expansion',
                'text': expanded_text,
                'description': 'Expandido con términos ambientales relevantes'
            })
        
        # Mejora 2: Reestructuración para claridad
        restructured_text = self._restructure_for_clarity(text, improvement_type)
        improvements.append({
            'version': 'restructured',
            'text': restructured_text,
            'description': 'Reestructurado para mayor claridad'
        })
        
        # Mejora 3: Optimización de longitud
        optimized_text = self._optimize_length(text, improvement_type)
        improvements.append({
            'version': 'length_optimized',
            'text': optimized_text,
            'description': 'Optimizado para longitud apropiada'
        })
        
        # Evaluar cada mejora usando BERT
        evaluated_improvements = []
        for improvement in improvements:
            improved_embeddings = self.generate_embeddings(improvement['text'])
            similarity = self.calculate_semantic_similarity(text, improvement['text'])
            
            evaluated_improvements.append({
                **improvement,
                'semantic_similarity': similarity['average_similarity'],
                'token_count': improved_embeddings['token_count'],
                'environmental_terms': len([
                    term for term in self.environmental_keywords 
                    if term.lower() in improvement['text'].lower()
                ])
            })
        
        # Seleccionar la mejor mejora
        best_improvement = max(
            evaluated_improvements, 
            key=lambda x: x['semantic_similarity'] + (x['environmental_terms'] * 0.1)
        )
        
        return {
            'original_text': text,
            'improvement_type': improvement_type,
            'environmental_focus': environmental_focus,
            'original_environmental_terms': environmental_terms_found,
            'all_improvements': evaluated_improvements,
            'best_improvement': best_improvement,
            'improvement_score': best_improvement['semantic_similarity']
        }
    
    def _expand_with_environmental_terms(self, text: str, existing_terms: List[str]) -> str:
        """Expande el texto con términos ambientales relevantes de manera más natural"""
        # Mapeo de términos relacionados
        term_expansions = {
            'medio ambiente': ['ecosistema', 'naturaleza', 'biodiversidad'],
            'sostenible': ['ecológico', 'verde', 'responsable'],
            'contaminación': ['emisiones', 'residuos', 'tóxicos'],
            'energía': ['renovable', 'limpia', 'eficiente'],
            'clima': ['calentamiento global', 'cambio climático'],
            'conservación': ['protección', 'preservación']
        }
        
        # Patrones de integración natural
        integration_patterns = [
            "Esto está estrechamente relacionado con {term}, un factor clave en este contexto.",
            "Considerando la importancia de {term} en este ámbito, es fundamental adoptar un enfoque integral.",
            "Este proceso contribuye significativamente a {term}, aspecto esencial para la sostenibilidad.",
            "La relación con {term} es innegable y debe ser considerada en cualquier estrategia ambiental.",
            "Un aspecto complementario es {term}, que potencia los efectos positivos de estas acciones."
        ]
        
        # Determinar si el texto termina con punto
        ends_with_period = text.rstrip().endswith('.')
        
        # Si no hay términos existentes, agregar algunos genéricos
        if not existing_terms:
            # Seleccionar términos genéricos para agregar
            generic_terms = ['sostenibilidad', 'conservación ambiental', 'responsabilidad ecológica']
            selected_term = np.random.choice(generic_terms)
            
            # Seleccionar un patrón de integración
            pattern = np.random.choice(integration_patterns)
            
            # Integrar el término
            if ends_with_period:
                expanded_text = f"{text} {pattern.format(term=selected_term)}"
            else:
                expanded_text = f"{text}. {pattern.format(term=selected_term)}"
            
            return expanded_text
        
        # Si hay términos existentes, expandir con términos relacionados
        expanded_text = text
        terms_added = 0
        
        for existing_term in existing_terms:
            if existing_term in term_expansions and terms_added < 2:  # Limitar a 2 términos adicionales
                for related_term in term_expansions[existing_term]:
                    if related_term.lower() not in text.lower():
                        # Seleccionar un patrón de integración
                        pattern = np.random.choice(integration_patterns)
                        
                        # Integrar el término relacionado
                        if ends_with_period or terms_added > 0:
                            expanded_text += f" {pattern.format(term=related_term)}"
                        else:
                            expanded_text += f". {pattern.format(term=related_term)}"
                        
                        terms_added += 1
                        break  # Solo agregar uno por término existente
        
        # Asegurar que el texto termine con punto
        if not expanded_text.rstrip().endswith('.'):
            expanded_text += '.'
            
        return expanded_text
    
    def _restructure_for_clarity(self, text: str, improvement_type: str) -> str:
        """Reestructura el texto para mayor claridad"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if improvement_type == 'titulo':
            # Para títulos, hacer más conciso y atractivo
            main_concept = sentences[0] if sentences else text
            return f"{main_concept.strip()}: Solución Ambiental Innovadora"
        
        elif improvement_type == 'descripcion':
            # Para descripciones, estructura problema-solución
            if len(sentences) >= 2:
                return f"{sentences[0]}. Esta situación requiere acción inmediata. {' '.join(sentences[1:])}."
            else:
                return f"{text} Esta iniciativa contribuye significativamente a la sostenibilidad ambiental."
        
        elif improvement_type == 'contenido':
            # Para contenido, agregar estructura y transiciones
            if len(sentences) >= 2:
                restructured = f"En primer lugar, {sentences[0].lower()}. "
                if len(sentences) > 2:
                    restructured += f"Además, {sentences[1].lower()}. "
                    restructured += f"Por último, {' '.join(sentences[2:]).lower()}."
                else:
                    restructured += f"En consecuencia, {sentences[1].lower()}."
                return restructured
            else:
                return f"Es importante destacar que {text.lower()} Esto representa un paso crucial hacia la sostenibilidad."
        
        return text
    
    def _optimize_length(self, text: str, improvement_type: str) -> str:
        """Optimiza la longitud del texto según el tipo"""
        words = text.split()
        
        target_lengths = {
            'titulo': (5, 12),
            'descripcion': (20, 50),
            'contenido': (50, 200),
            'llamada_accion': (10, 25),
            'resumen': (15, 40)
        }
        
        min_len, max_len = target_lengths.get(improvement_type, (20, 100))
        
        if len(words) < min_len:
            # Expandir texto corto
            if improvement_type == 'titulo':
                return f"{text}: Innovación para el Futuro Sostenible"
            else:
                return f"{text} Esta iniciativa representa un avance significativo en la protección ambiental y el desarrollo sostenible."
        
        elif len(words) > max_len:
            # Acortar texto largo
            if improvement_type == 'titulo':
                # Tomar las primeras palabras clave
                key_words = words[:8]
                return ' '.join(key_words)
            else:
                # Resumir manteniendo ideas principales
                sentences = text.split('.')
                main_sentences = sentences[:2] if len(sentences) > 2 else sentences
                return '. '.join(main_sentences).strip() + '.'
        
        return text
    
    def generate_variations(self, text: str, num_variations: int = 3) -> Dict:
        """
        Genera múltiples variaciones del texto
        """
        variations = []
        
        # Variación 1: Enfoque técnico
        technical_variation = self._create_technical_variation(text)
        variations.append({
            'type': 'technical',
            'text': technical_variation,
            'description': 'Enfoque técnico y profesional'
        })
        
        # Variación 2: Enfoque emocional
        emotional_variation = self._create_emotional_variation(text)
        variations.append({
            'type': 'emotional',
            'text': emotional_variation,
            'description': 'Enfoque emocional y persuasivo'
        })
        
        # Variación 3: Enfoque educativo
        educational_variation = self._create_educational_variation(text)
        variations.append({
            'type': 'educational',
            'text': educational_variation,
            'description': 'Enfoque educativo e informativo'
        })
        
        # Evaluar cada variación
        evaluated_variations = []
        for variation in variations:
            similarity = self.calculate_semantic_similarity(text, variation['text'])
            embeddings = self.generate_embeddings(variation['text'])
            
            evaluated_variations.append({
                **variation,
                'semantic_similarity': similarity['average_similarity'],
                'token_count': embeddings['token_count'],
                'environmental_score': self._calculate_environmental_score(variation['text'])
            })
        
        return {
            'original_text': text,
            'variations': evaluated_variations,
            'best_variation': max(evaluated_variations, key=lambda x: x['environmental_score'])
        }
    
    def _create_technical_variation(self, text: str) -> str:
        """Crea una variación con enfoque técnico"""
        technical_terms = {
            'problema': 'desafío técnico',
            'solución': 'implementación estratégica',
            'importante': 'crítico',
            'bueno': 'eficiente',
            'malo': 'ineficiente',
            'ayuda': 'optimiza',
            'hace': 'ejecuta',
            'permitido': 'facilitado',
            'comenzar': 'iniciar',
            'concientizar': 'generar conciencia colectiva',
            'acciones': 'intervenciones estratégicas',
            'ambientales': 'ecosistémicas',
            'cuidar': 'preservar sistemáticamente',
            'cuidemos': 'preservemos sistemáticamente',
            'planeta': 'ecosistema global'
        }
        
        # Aplicar reemplazos técnicos
        technical_text = text.lower()
        for original, technical in technical_terms.items():
            technical_text = technical_text.replace(original, technical)
        
        # Eliminar puntos finales para evitar duplicados
        technical_text = technical_text.rstrip('.')
        
        # Asegurar que el texto tenga sentido gramatical
        if len(technical_text.split()) <= 3:
            technical_text = "la necesidad de " + technical_text
        
        # Diferentes formatos según longitud
        words = text.split()
        if len(words) > 10:
            result = "Los análisis técnicos demuestran que " + technical_text + ". Este fenómeno ha sido documentado mediante indicadores de sostenibilidad."
        else:
            result = "El análisis sistemático indica que " + technical_text + ". Este paradigma garantiza resultados cuantificables y sostenibles."
        
        # Asegurar capitalización correcta después de puntos
        result = '. '.join(s.strip().capitalize() for s in result.split('.') if s.strip())
        return result + "."
    
    def _create_emotional_variation(self, text: str) -> str:
        """Crea una variación con enfoque emocional"""
        # Términos emocionales para reemplazar
        emotional_terms = {
            'importante': 'vital',
            'necesario': 'urgente',
            'bueno': 'extraordinario',
            'problema': 'crisis',
            'cambio': 'transformación',
            'mejorar': 'revolucionar',
            'ayudar': 'salvar',
            'proteger': 'defender apasionadamente',
            'ambiente': 'hogar natural',
            'planeta': 'único y frágil planeta',
            'cuidar': 'salvar',
            'cuidemos': 'salvemos juntos',
            'debemos': 'tenemos la responsabilidad de'
        }
        
        # Aplicar reemplazos emocionales
        emotional_text = text.lower()
        for original, emotional in emotional_terms.items():
            emotional_text = emotional_text.replace(original, emotional)
        
        # Eliminar puntos finales para evitar duplicados
        emotional_text = emotional_text.rstrip('.')
        
        # Asegurar que el texto tenga sentido gramatical para entradas muy cortas
        if len(emotional_text.split()) <= 3:
            emotional_text = "comprometernos con " + emotional_text
        
        # Diferentes formatos según longitud
        words = text.split()
        if len(words) < 10:
            result = "¡Es momento de actuar! " + emotional_text.capitalize() + ". ¿Te unirás a esta misión vital para proteger nuestro futuro?"
        else:
            result = "Cada día que pasa sin actuar es una oportunidad perdida. " + emotional_text + ". Esta es nuestra responsabilidad compartida con las generaciones futuras."
        
        # Asegurar capitalización correcta después de puntos
        result = '. '.join(s.strip().capitalize() for s in result.split('.') if s.strip())
        return result + "."
    
    def _create_educational_variation(self, text: str) -> str:
        """Crea una variación con enfoque educativo"""
        # Términos educativos para reemplazar
        educational_terms = {
            'importante': 'fundamental para el ecosistema',
            'problema': 'fenómeno ambiental',
            'cambio': 'proceso de transformación',
            'contaminación': 'degradación ambiental',
            'proteger': 'conservar la biodiversidad de',
            'cuidar': 'preservar y proteger',
            'cuidemos': 'debemos preservar y proteger',
            'planeta': 'ecosistema terrestre'
        }
        
        # Aplicar reemplazos educativos
        educational_text = text.lower()
        for original, educational in educational_terms.items():
            educational_text = educational_text.replace(original, educational)
        
        # Eliminar puntos finales para evitar duplicados
        educational_text = educational_text.rstrip('.')
        
        # Asegurar que el texto tenga sentido gramatical para entradas muy cortas
        if len(educational_text.split()) <= 3:
            educational_text = "la importancia de " + educational_text + " para la sostenibilidad"
        
        # Diferentes formatos según longitud
        words = text.split()
        if len(words) > 15:
            result = "Un aspecto fundamental en la educación ambiental es entender que " + educational_text + ". Los estudios científicos han documentado cómo este conocimiento contribuye a formar ciudadanos ambientalmente responsables."
        else:
            result = "¿Sabías que " + educational_text + "? Este es un concepto clave en la educación ambiental moderna. Comprender estos principios es esencial para desarrollar una relación sostenible con nuestro entorno."
        
        # Asegurar capitalización correcta después de puntos
        result = '. '.join(s.strip().capitalize() for s in result.split('.') if s.strip())
        return result + "."
    
    def _calculate_environmental_score(self, text: str) -> float:
        """Calcula un score de relevancia ambiental"""
        score = 0
        text_lower = text.lower()
        
        for keyword in self.environmental_keywords:
            if keyword in text_lower:
                score += 1
        
        # Normalizar por longitud del texto
        words = len(text.split())
        normalized_score = (score / words) * 100 if words > 0 else 0
        
        return min(normalized_score, 10.0)  # Máximo 10
    
    def comprehensive_text_improvement(self, text: str, target_type: str = 'contenido', 
                                     options: Dict = None) -> Dict:
        """
        Mejora completa del texto usando todas las capacidades de BERT
        """
        if options is None:
            options = {
                'environmental_focus': True,
                'generate_variations': True,
                'optimize_length': True,
                'include_embeddings': False,
                'similarity_threshold': 0.7
            }
        
        # Análisis inicial
        original_embeddings = self.generate_embeddings(text) if options.get('include_embeddings') else None
        
        # Mejora principal
        main_improvement = self.improve_text_with_context(
            text, 
            target_type, 
            options.get('environmental_focus', True)
        )
        
        # Generar variaciones si se solicita
        variations = None
        if options.get('generate_variations', True):
            variations = self.generate_variations(main_improvement['best_improvement']['text'])
        
        # Resultado final
        final_text = variations['best_variation']['text'] if variations else main_improvement['best_improvement']['text']
        
        return {
            'original_text': text,
            'target_type': target_type,
            'options': options,
            'main_improvement': main_improvement,
            'variations': variations,
            'final_improved_text': final_text,
            'improvement_summary': {
                'original_length': len(text.split()),
                'final_length': len(final_text.split()),
                'environmental_terms_added': self._calculate_environmental_score(final_text) - self._calculate_environmental_score(text),
                'semantic_preservation': main_improvement['improvement_score'],
                'coherence_score': 0.8,  # Valor predeterminado
                'grammar_score': 0.9     # Valor predeterminado
            },
            'original_embeddings': original_embeddings
        }
    
    def comprehensive_text_improvement_enhanced(self, text: str, target_type: str = 'contenido', 
                                              options: Dict = None) -> Dict:
        """
        Versión mejorada del procesamiento completo de texto
        """
        result = self.comprehensive_text_improvement(text, target_type, options)
        
        # Añadir métricas adicionales para compatibilidad con el sistema completo
        result['improvement_summary']['coherence_score'] = 0.8
        result['improvement_summary']['grammar_score'] = 0.9
        
        return result