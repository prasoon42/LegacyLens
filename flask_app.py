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
        
        # Preprocess first (consistent with backend)
        img = preprocess_image(img)
        
        # Detect
        items = detect_text(img)
        
        # Visualize
        vis_img = visualize(img, items)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        vis_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'items': items,
            'visualization': vis_b64
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
