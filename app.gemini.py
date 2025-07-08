from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
import uuid
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API - Add your API key here
GEMINI_API_KEY = 'API KEY'  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class GeminiMathSolver:
    def __init__(self):
        self.model = model
    
    def process_image_and_solve(self, image_path):
        """Process image using Gemini Vision API and solve math"""
        try:
            # Read the image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # Convert to PIL Image for Gemini
            image = Image.open(BytesIO(image_data))
            
            # Prepare the prompt for Gemini
            prompt = """
            You are a mathematical problem solver. Please analyze this handwritten math problem image and:
            
            1. Extract the mathematical expression or equation from the image
            2. Identify the type of problem (arithmetic, algebra, calculus, etc.)
            3. Solve the problem step by step
            4. Provide the final answer
            
            Please respond in the following JSON format:
            {
                "raw_ocr_text": "what you see written in the image",
                "cleaned_expression": "the mathematical expression in proper notation",
                "expression_type": "type of math problem",
                "solution": "final answer",
                "steps": ["step 1", "step 2", "step 3", ...],
                "error": null
            }
            
            If you cannot read the image or solve the problem, set "error" to describe the issue.
            """
            
            # Send to Gemini
            response = self.model.generate_content([prompt, image])
            
            # Parse response
            response_text = response.text
            logger.info(f"Gemini response: {response_text}")
            
            # Try to extract JSON from response
            try:
                import json
                import re
                
                # Find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # If no JSON found, create a structured response
                    result = {
                        "raw_ocr_text": response_text,
                        "cleaned_expression": "Could not extract expression",
                        "expression_type": "unknown",
                        "solution": response_text,
                        "steps": ["See solution above"],
                        "error": None
                    }
                
                return result
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return {
                    "raw_ocr_text": response_text,
                    "cleaned_expression": "Could not parse structured response",
                    "expression_type": "unknown",
                    "solution": response_text,
                    "steps": ["See full response above"],
                    "error": None
                }
        
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            return {
                "raw_ocr_text": "",
                "cleaned_expression": "",
                "expression_type": "error",
                "solution": None,
                "steps": [],
                "error": str(e)
            }
    
    def solve_expression(self, expression):
        """Solve a text mathematical expression using Gemini"""
        try:
            prompt = f"""
            Please solve this mathematical expression step by step: {expression}
            
            Provide your response in the following JSON format:
            {{
                "cleaned_expression": "{expression}",
                "expression_type": "type of math problem",
                "solution": "final answer",
                "steps": ["step 1", "step 2", "step 3", ...],
                "error": null
            }}
            
            If you cannot solve the problem, set "error" to describe the issue.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON from response
            try:
                import json
                import re
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {
                        "cleaned_expression": expression,
                        "expression_type": "unknown",
                        "solution": response_text,
                        "steps": ["See solution above"],
                        "error": None
                    }
                
                return result
                
            except json.JSONDecodeError:
                return {
                    "cleaned_expression": expression,
                    "expression_type": "unknown",
                    "solution": response_text,
                    "steps": ["See full response above"],
                    "error": None
                }
        
        except Exception as e:
            logger.error(f"Error solving expression with Gemini: {str(e)}")
            return {
                "cleaned_expression": expression,
                "expression_type": "error",
                "solution": None,
                "steps": [],
                "error": str(e)
            }

# Initialize the Gemini math solver
solver = GeminiMathSolver()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_math():
    """
    API endpoint to solve handwritten math from uploaded image using Gemini
    
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
        
        # Check if API key is configured
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            return jsonify({
                'error': 'Gemini API key not configured. Please set your API key in the code.',
                'success': False
            }), 500
        
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
        
        logger.info(f"Processing image with Gemini: {filename}")
        
        # Process image and solve using Gemini
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
    API endpoint to solve math expression from text input using Gemini
    
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
        
        # Check if API key is configured
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            return jsonify({
                'error': 'Gemini API key not configured. Please set your API key in the code.',
                'success': False
            }), 500
        
        expression = data['expression']
        logger.info(f"Solving text expression with Gemini: {expression}")
        
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
        'message': 'Gemini Math Solver API is running',
        'version': '1.0.0',
        'gemini_configured': GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
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
    
    print("🚀 Starting Gemini Math Solver API...")
    print("🤖 Using Google Gemini for handwriting recognition")
    print("📋 Available endpoints:")
    print("  GET  /              - Main web interface")
    print("  POST /api/solve     - Solve from image (Gemini)")
    print("  POST /api/solve-text - Solve from text (Gemini)")
    print("  GET  /api/health    - Health check")
    print("🌐 Access the app at: http://localhost:5000")
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️  WARNING: Please set your Gemini API key in the code!")
    else:
        print("✅ Gemini API key configured")
    
    app.run(debug=True, host='0.0.0.0', port=5000)