from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from math_solver_backend import MathSolver  # Fixed import
import tempfile
import uuid
import logging
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the math solver
solver = MathSolver()

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_math():
    """
    API endpoint to solve handwritten math from uploaded image
    
    Expected JSON format:
    {
        "image": "base64_encoded_image_string",
        "format": "jpg|png|jpeg"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'success': False
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode and save image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save image
        img.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        
        # Process image and solve
        result = solver.process_image_and_solve(filepath)
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except:
            logger.warning(f"Could not remove temporary file: {filepath}")
        
        # Format solution for display
        solution_display = result.get('solution', None)
        if solution_display is not None:
            if isinstance(solution_display, (list, tuple)):
                solution_display = str(solution_display)
            elif hasattr(solution_display, '__str__'):
                solution_display = str(solution_display)
        
        # Format response
        response = {
            'success': True,
            'result': result,
            'original_expression': result.get('cleaned_expression', ''),
            'solution': solution_display,
            'steps': result.get('steps', []),
            'expression_type': result.get('expression_type', 'unknown'),
            'raw_ocr_text': result.get('raw_ocr_text', ''),
            'error': result.get('error', None)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/solve-text', methods=['POST'])
def solve_text():
    """
    API endpoint to solve math expression from text input
    
    Expected JSON format:
    {
        "expression": "mathematical_expression"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'expression' not in data:
            return jsonify({
                'error': 'No expression provided',
                'success': False
            }), 400
        
        expression = data['expression']
        logger.info(f"Solving text expression: {expression}")
        
        result = solver.solve_expression(expression)
        
        # Format solution for display
        solution_display = result.get('solution', None)
        if solution_display is not None:
            if isinstance(solution_display, (list, tuple)):
                solution_display = str(solution_display)
            elif hasattr(solution_display, '__str__'):
                solution_display = str(solution_display)
        
        response = {
            'success': True,
            'result': result,
            'original_expression': expression,
            'solution': solution_display,
            'steps': result.get('steps', []),
            'expression_type': result.get('expression_type', 'unknown'),
            'error': result.get('error', None)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error solving text expression: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Math Solver API is running',
        'version': '1.0.0'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    # Create required directories
    directories = ['templates', 'static', 'uploads']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    print("🚀 Starting Math Solver API...")
    print("📋 Available endpoints:")
    print("  GET  /              - Main web interface")
    print("  POST /api/solve     - Solve from image")
    print("  POST /api/solve-text - Solve from text")
    print("  GET  /api/health    - Health check")
    print("🌐 Access the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
