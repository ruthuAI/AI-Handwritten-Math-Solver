import cv2
import numpy as np
import pytesseract
import sympy as sp
from sympy import symbols, solve, simplify, expand, factor
import re
from typing import Optional, Tuple, List
import base64
from io import BytesIO
from PIL import Image

class MathSolver:
    def __init__(self):
        """Initialize the Math Solver with OCR and symbolic math capabilities"""
        self.x, self.y, self.z = symbols('x y z')
        self.variables = [self.x, self.y, self.z]
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the image for better OCR accuracy
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize image for better OCR (if too small)
        height, width = cleaned.shape
        if height < 100 or width < 100:
            scale_factor = max(100 / height, 100 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def extract_text_from_image(self, processed_image: np.ndarray) -> str:
        """
        Extract text from preprocessed image using OCR
        
        Args:
            processed_image: Preprocessed image
            
        Returns:
            Extracted text string
        """
        # Configure Tesseract for math expressions
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()^xyzXYZ.'
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        return text.strip()
    
    def clean_and_parse_expression(self, raw_text: str) -> str:
        """
        Clean and parse the extracted text into a valid mathematical expression
        
        Args:
            raw_text: Raw text from OCR
            
        Returns:
            Cleaned mathematical expression
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', raw_text.strip())
        
        # Common OCR corrections
        corrections = {
            'O': '0',
            'o': '0',
            'I': '1',
            'l': '1',
            'S': '5',
            'G': '6',
            'B': '8',
            'g': '9',
            'x': '*',  # Sometimes multiplication is recognized as x
            'X': '*',
            '×': '*',
            '÷': '/',
            '−': '-',
            '=': '=',
            '^': '**',  # Convert to Python power notation
        }
        
        # Apply corrections
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Add implicit multiplication (e.g., "2x" -> "2*x")
        text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
        text = re.sub(r'(\))(\d)', r'\1*\2', text)
        text = re.sub(r'(\d)(\()', r'\1*\2', text)
        
        return text
    
    def solve_expression(self, expression: str) -> dict:
        """
        Solve the mathematical expression
        
        Args:
            expression: Mathematical expression to solve
            
        Returns:
            Dictionary containing solution details
        """
        result = {
            'original_expression': expression,
            'solution': None,
            'steps': [],
            'error': None,
            'expression_type': 'unknown'
        }
        
        try:
            # Split on equals sign if present (equation)
            if '=' in expression:
                parts = expression.split('=')
                if len(parts) == 2:
                    left_expr = sp.sympify(parts[0].strip())
                    right_expr = sp.sympify(parts[1].strip())
                    equation = sp.Eq(left_expr, right_expr)
                    
                    # Solve the equation
                    solutions = solve(equation, self.variables)
                    result['solution'] = solutions
                    result['expression_type'] = 'equation'
                    result['steps'].append(f"Original equation: {equation}")
                    result['steps'].append(f"Solutions: {solutions}")
                    
            else:
                # Simple expression evaluation
                expr = sp.sympify(expression)
                
                # Check if it's a polynomial that can be factored
                if expr.is_polynomial():
                    factored = factor(expr)
                    result['steps'].append(f"Factored form: {factored}")
                
                # Try to simplify
                simplified = simplify(expr)
                result['solution'] = simplified
                result['expression_type'] = 'expression'
                result['steps'].append(f"Simplified: {simplified}")
                
                # If it's a numeric expression, evaluate it
                if expr.is_number:
                    result['solution'] = float(expr)
                    result['expression_type'] = 'arithmetic'
                
        except Exception as e:
            result['error'] = str(e)
            result['steps'].append(f"Error: {str(e)}")
        
        return result
    
    def process_image_and_solve(self, image_path: str) -> dict:
        """
        Complete pipeline: preprocess image, extract text, and solve
        
        Args:
            image_path: Path to input image
            
        Returns:
            Complete solution dictionary
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Extract text
            raw_text = self.extract_text_from_image(processed_img)
            
            # Clean and parse expression
            cleaned_expr = self.clean_and_parse_expression(raw_text)
            
            # Solve the expression
            result = self.solve_expression(cleaned_expr)
            
            # Add OCR details
            result['raw_ocr_text'] = raw_text
            result['cleaned_expression'] = cleaned_expr
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'solution': None,
                'steps': [f"Processing error: {str(e)}"]
            }

# Example usage and testing
if __name__ == "__main__":
    solver = MathSolver()
    
    # Test with sample expressions
    test_expressions = [
        "2x + 5 = 13",
        "x^2 - 4x + 3 = 0",
        "3*4 + 2*5",
        "sin(x) + cos(x)",
        "2x^2 + 3x - 1"
    ]
    
    print("Testing Math Solver with sample expressions:")
    print("=" * 50)
    
    for expr in test_expressions:
        result = solver.solve_expression(expr)
        print(f"\nExpression: {expr}")
        print(f"Type: {result['expression_type']}")
        print(f"Solution: {result['solution']}")
        if result['steps']:
            print("Steps:")
            for step in result['steps']:
                print(f"  - {step}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("-" * 30)