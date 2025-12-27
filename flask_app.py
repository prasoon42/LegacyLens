#!/usr/bin/env python3
"""Generic 7-Segment OCR Web App"""
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.detect_and_ocr import detect_text, preprocess_image, visualize
from src.ai_interpreter import ai_interpreter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Debug: Save upload
        is_debug = request.form.get('debug') == 'true'
        if is_debug:
            debug_dir = os.path.join(os.path.dirname(__file__), 'debug_uploads')
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(cv2.getTickCount())
            cv2.imwrite(os.path.join(debug_dir, f'upload_{timestamp}.jpg'), img)
        
        # Preprocess first (consistent with backend)
        img = preprocess_image(img)
        
        # Detect
        items = detect_text(img)
        
        # Visualize
        vis_img = visualize(img, items)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        vis_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # AI Interpretation
        ai_result = None
        if items:
            # Combine all detected text for interpretation
            # For now, take the highest confidence or first item
            # In a real scenario, we might want to interpret specific fields
            primary_reading = items[0]['text']
            
            # Get AI interpretation
            try:
                ai_result = ai_interpreter.interpret_reading(
                    device_id='default_camera', 
                    reading=primary_reading,
                    device_type='legacy_display'
                )
            except Exception as e:
                print(f"AI Error: {e}")
        
        return jsonify({
            'success': True,
            'items': items,
            'visualization': vis_b64,
            'ai_interpretation': ai_result
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/interpret', methods=['POST'])
def interpret():
    """Interpret OCR results using GenAI"""
    try:
        data = request.get_json()
        
        if not data or 'reading' not in data:
            return jsonify({'error': 'Missing reading data'}), 400
        
        reading = data['reading']
        device_id = data.get('device_id', 'default')
        device_type = data.get('device_type', '7-segment display')
        
        # Get AI interpretation
        result = ai_interpreter.interpret_reading(device_id, reading, device_type)
        
        return jsonify({
            'success': True,
            'interpretation': result
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question about a device"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question'}), 400
        
        question = data['question']
        device_id = data.get('device_id', 'default')
        
        # Get AI answer
        answer = ai_interpreter.ask_question(device_id, question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
