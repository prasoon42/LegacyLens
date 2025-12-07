#!/usr/bin/env python3
"""Flask-based OCR web application - no caching issues!"""
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.detect_and_ocr import preprocess_image, detect_digits, map_rows_to_metrics, visualize

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max

# Store readings in memory (in production, use a database)
readings_history = []

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
        
        # Process image
        img = preprocess_image(img)
        boxes = detect_digits(img)
        metrics, debug_img = map_rows_to_metrics(img, boxes)
        
        # Create visualization
        vis_img = visualize(img, boxes, metrics)
        
        # Convert images to base64 for web display
        _, buffer = cv2.imencode('.jpg', vis_img)
        vis_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, debug_buffer = cv2.imencode('.jpg', debug_img)
        debug_b64 = base64.b64encode(debug_buffer).decode('utf-8')
        
        # Store in history
        reading = {
            'timestamp': datetime.now().strftime('%H:%M'),
            'sys': metrics.get('SYS', '--'),
            'dia': metrics.get('DIA', '--'),
            'pulse': metrics.get('PULSE', '--')
        }
        readings_history.insert(0, reading)
        if len(readings_history) > 10:
            readings_history.pop()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'visualization': vis_b64,
            'debug': debug_b64,
            'history': readings_history
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    return jsonify({'history': readings_history})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
