from flask import Flask, request, jsonify
import json
import numpy as np
import os
import threading
from modules.text_mining_system import TextMiningSystem

app = Flask(__name__)

# Inicialización lazy del sistema
text_mining_system = None
system_lock = threading.Lock()

def get_text_mining_system():
    """Obtiene el sistema de minería de texto con inicialización lazy"""
    global text_mining_system
    if text_mining_system is None:
        with system_lock:
            if text_mining_system is None:
                print("🚀 Inicializando sistema de minería de texto...")
                text_mining_system = TextMiningSystem()
                print("✅ Sistema inicializado")
    return text_mining_system

# Clase auxiliar para serializar arrays numpy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/api/process_text', methods=['POST'])
def process_text():
    """
    Endpoint principal para procesar texto con el sistema completo
    """
    try:
        # Obtener datos del request
        data = request.get_json()
        
        # Validar entrada
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        content_type = data.get('content_type', 'contenido')
        track_steps = data.get('track_steps', False)
        
        # Respuesta rápida para textos muy largos
        if len(text) > 500:
            return jsonify({
                'error': 'Texto demasiado largo. Máximo 500 caracteres para evitar timeouts.',
                'suggestion': 'Usa el endpoint /api/process_text_simple para textos largos'
            }), 400
        
        # Obtener sistema con inicialización lazy
        system = get_text_mining_system()
        
        # Procesar el texto con timeout implícito
        result = system.process_text_complete_enhanced(
            text=text,
            content_type=content_type,
            track_steps=track_steps
        )
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_text_simple', methods=['POST'])
def process_text_simple():
    """
    Endpoint simplificado sin BERT para textos largos o respuesta rápida
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        content_type = data.get('content_type', 'contenido')
        
        # Procesamiento básico paso a paso SIN BERT
        try:
            # Obtener sistema con inicialización lazy
            system = get_text_mining_system()
            
            # Paso 1: Ingesta
            ingestion_result = system.ingestion.ingest_manual_text(text)
            current_text = ingestion_result['original_text']
            
            # Paso 2: Limpieza básica
            cleaning_options = {
                'remove_urls': True,
                'remove_emails': True,
                'remove_phones': True,
                'remove_html': True,
                'remove_special_chars': False,
                'keep_punctuation': True,
                'normalize_whitespace': True,
                'normalize_punctuation': True,
                'remove_newlines': True,
                'fix_encoding': True
            }
            cleaning_result = system.cleaner.basic_clean(current_text, cleaning_options)
            current_text = cleaning_result['cleaned_text']
            
            # Paso 3: Tokenización simple
            tokens = current_text.split()  # Tokenización básica sin spaCy
            
            # Paso 4: Normalización básica
            normalized_tokens = [token.lower().strip() for token in tokens if token.strip()]
            current_text = ' '.join(normalized_tokens)
            
            # Resultado final
            result = {
                'original_text': text,
                'final_text': current_text,
                'content_type': content_type,
                'processing_type': 'simple_complete',
                'steps_completed': ['ingesta', 'limpieza', 'tokenización', 'normalización'],
                'message': 'Procesamiento completo sin BERT - Rápido y funcional',
                'word_count': len(normalized_tokens),
                'character_count': len(current_text)
            }
            
            return jsonify(result), 200
            
        except Exception as processing_error:
            # Fallback a procesamiento ultra-básico
            result = {
                'original_text': text,
                'final_text': text.strip().lower(),
                'processing_type': 'basic_fallback',
                'message': f'Procesamiento básico de fallback: {str(processing_error)}'
            }
            return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_text_no_bert', methods=['POST'])
def process_text_no_bert():
    """
    Endpoint que ejecuta todos los pasos EXCEPTO BERT
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        content_type = data.get('content_type', 'contenido')
        track_steps = data.get('track_steps', False)
        
        # Límite de seguridad
        if len(text) > 1000:
            return jsonify({
                'error': 'Texto demasiado largo. Máximo 1000 caracteres.',
                'suggestion': 'Usa /api/process_text_simple para procesamiento básico'
            }), 400
        
        intermediate_results = {}
        processing_steps = []
        
        try:
            # Obtener sistema con inicialización lazy
            system = get_text_mining_system()
            
            # Paso 1: Ingesta
            ingestion_result = system.ingestion.ingest_manual_text(text)
            current_text = ingestion_result['original_text']
            
            if track_steps:
                intermediate_results['ingestion'] = current_text
                processing_steps.append({'step': 'ingesta', 'status': 'completed'})
            
            # Paso 2: Limpieza
            cleaning_options = {
                'remove_urls': True,
                'remove_emails': True,
                'remove_phones': True,
                'remove_html': True,
                'remove_special_chars': False,
                'keep_punctuation': True,
                'normalize_whitespace': True,
                'normalize_punctuation': True,
                'remove_newlines': True,
                'fix_encoding': True
            }
            cleaning_result = system.cleaner.basic_clean(current_text, cleaning_options)
            current_text = cleaning_result['cleaned_text']
            
            if track_steps:
                intermediate_results['cleaning'] = current_text
                processing_steps.append({'step': 'limpieza', 'status': 'completed'})
            
            # Paso 3: Tokenización (sin spaCy para evitar problemas)
            tokens = current_text.split()
            
            if track_steps:
                intermediate_results['tokenization'] = ' '.join(tokens[:10]) + ('...' if len(tokens) > 10 else '')
                processing_steps.append({'step': 'tokenización', 'status': 'completed', 'token_count': len(tokens)})
            
            # Paso 4: Normalización básica
            normalized_tokens = []
            for token in tokens:
                # Normalización simple sin librerías pesadas
                normalized_token = token.lower().strip()
                if len(normalized_token) > 2:  # Filtrar palabras muy cortas
                    normalized_tokens.append(normalized_token)
            
            current_text = ' '.join(normalized_tokens)
            
            if track_steps:
                intermediate_results['normalization'] = current_text
                processing_steps.append({'step': 'normalización', 'status': 'completed'})
            
            # Resultado final sin BERT
            result = {
                'original_text': text,
                'final_text': current_text,
                'content_type': content_type,
                'processing_type': 'complete_no_bert',
                'intermediate_results': intermediate_results if track_steps else {},
                'processing_steps': processing_steps if track_steps else [],
                'steps_completed': ['ingesta', 'limpieza', 'tokenización', 'normalización'],
                'steps_skipped': ['eliminación_ruido', 'lematización', 'bert_processing'],
                'message': 'Procesamiento completo sin BERT - Optimizado para Railway',
                'metrics': {
                    'original_word_count': len(text.split()),
                    'final_word_count': len(normalized_tokens),
                    'character_reduction': len(text) - len(current_text)
                }
            }
            
            return jsonify(result), 200
            
        except Exception as processing_error:
            return jsonify({
                'error': f'Error en procesamiento: {str(processing_error)}',
                'fallback_result': {
                    'original_text': text,
                    'final_text': text.strip().lower(),
                    'processing_type': 'error_fallback'
                }
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bert/improve', methods=['POST'])
def bert_improve():
    """
    Endpoint específico para mejora de texto con BERT
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        improvement_type = data.get('improvement_type', 'contenido')
        environmental_focus = data.get('environmental_focus', True)
        
        # Obtener sistema con inicialización lazy
        system = get_text_mining_system()
        
        # Mejorar texto con BERT
        result = system.bert_processor.improve_text_with_context(
            text=text,
            improvement_type=improvement_type,
            environmental_focus=environmental_focus
        )
        
        return json.dumps(result, cls=NumpyEncoder), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bert/generate_variations', methods=['POST'])
def bert_generate_variations():
    """
    Endpoint para generar variaciones de texto con BERT
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        num_variations = data.get('num_variations', 3)
        
        # Obtener sistema con inicialización lazy
        system = get_text_mining_system()
        
        # Generar variaciones
        result = system.bert_processor.generate_variations(
            text=text,
            num_variations=num_variations
        )
        
        return json.dumps(result, cls=NumpyEncoder), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bert/embeddings', methods=['POST'])
def bert_embeddings():
    """
    Endpoint para generar embeddings de texto con BERT
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'El campo "text" es requerido'}), 400
        
        text = data['text']
        
        # Obtener sistema con inicialización lazy
        system = get_text_mining_system()
        
        # Generar embeddings
        result = system.bert_processor.generate_embeddings(text)
        
        # Usar el encoder personalizado para manejar arrays numpy
        return json.dumps(result, cls=NumpyEncoder), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar que el servidor está funcionando
    """
    return jsonify({
        'status': 'ok',
        'message': 'El servidor de mejora de texto está funcionando correctamente',
        'timestamp': str(np.datetime64('now'))
    }), 200

@app.route('/', methods=['GET'])
def root():
    """
    Endpoint raíz con información básica
    """
    return jsonify({
        'service': 'Text Mining System API',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'warmup': '/warmup',
            'process_text': '/api/process_text (máx 500 chars)',
            'process_text_simple': '/api/process_text_simple (sin BERT)',
            'bert_improve': '/api/bert/improve',
            'bert_variations': '/api/bert/generate_variations',
            'bert_embeddings': '/api/bert/embeddings'
        }
    }), 200

@app.route('/warmup', methods=['GET'])
def warmup():
    """
    Endpoint ligero para verificar componentes básicos
    """
    try:
        print("🔥 Verificando componentes básicos...")
        
        # Obtener sistema con inicialización lazy
        system = get_text_mining_system()
        
        # Solo verificar que los módulos se pueden importar
        status = {
            'ingestion': hasattr(system, 'ingestion'),
            'cleaner': hasattr(system, 'cleaner'),
            'tokenizer': hasattr(system, 'tokenizer'),
            'bert_available': hasattr(system, 'bert_processor')
        }
        
        return jsonify({
            'status': 'ready',
            'message': 'Componentes básicos verificados',
            'components': status,
            'note': 'BERT se cargará bajo demanda para evitar timeouts'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error en verificación: {str(e)}'
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)