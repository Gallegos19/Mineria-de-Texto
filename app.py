from flask import Flask, request, jsonify
import json
import numpy as np
from modules.text_mining_system import TextMiningSystem

app = Flask(__name__)
text_mining_system = TextMiningSystem()

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
        
        # Procesar el texto con timeout implícito
        result = text_mining_system.process_text_complete_enhanced(
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
        
        # Procesamiento básico sin BERT
        result = {
            'original_text': text,
            'final_text': text.strip().lower(),
            'processing_type': 'simple',
            'message': 'Procesamiento básico completado sin BERT para evitar timeouts'
        }
        
        return jsonify(result), 200
    
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
        
        # Mejorar texto con BERT
        result = text_mining_system.bert_processor.improve_text_with_context(
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
        
        # Generar variaciones
        result = text_mining_system.bert_processor.generate_variations(
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
        
        # Generar embeddings
        result = text_mining_system.bert_processor.generate_embeddings(text)
        
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
    Endpoint para precargar modelos y evitar timeouts en la primera petición
    """
    try:
        print("🔥 Iniciando warm-up de modelos...")
        
        # Precargar BERT
        text_mining_system.bert_processor._load_bert_model()
        
        # Procesar un texto de prueba pequeño
        test_result = text_mining_system.process_text_complete_enhanced(
            text="test",
            content_type="titulo",
            track_steps=False
        )
        
        return jsonify({
            'status': 'warmed_up',
            'message': 'Modelos precargados exitosamente',
            'bert_loaded': text_mining_system.bert_processor.model is not None,
            'test_processed': test_result is not None
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'partial_warmup',
            'message': f'Warm-up parcial: {str(e)}',
            'note': 'La aplicación funcionará pero puede ser más lenta en la primera petición'
        }), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)