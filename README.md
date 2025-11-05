# Handwritten Mathematical Expression Recognition

This project is a web-based application that recognizes handwritten mathematical expressions from images or canvas drawings and converts them into LaTeX format or solves them. It leverages deep learning with Convolutional Neural Networks (CNNs) for symbol recognition and provides a user-friendly interface for uploading images or drawing equations directly in the browser.

## Features
- **Image Upload**: Upload PNG images of handwritten mathematical expressions.
- **Canvas Drawing**: Draw equations directly on a web-based canvas.
- **Symbol Recognition**: Identifies digits (0-9), operators (+, -, *, /, =), and variables (x, y).
- **Output Formats**: Outputs recognized expressions in LaTeX or as solved results (e.g., `2 + 3 = 5`).
- **Preprocessing**: Includes noise removal, binarization, and symbol segmentation for accurate recognition.
- **Web Interface**: Built with Flask for seamless interaction.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- A modern web browser (e.g., Chrome, Firefox, Edge)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Basavask/image-math-hand-written-recong.git
   cd image-math-hand-written-recong
   git checkout develop
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   python src/app.py
   ```
4. **Access the Web Interface**:
   - Open your browser and navigate to `http://localhost:5000`.

## Usage
1. **Upload an Image**:
   - On the homepage, click "Choose File" to select a PNG image of a handwritten equation.
   - Ensure the image is <50kb, with black symbols on a white background and clear separation between symbols.
   - Click "Upload" to process the image.
2. **Draw an Equation**:
   - Use the canvas tool on the homepage to draw your equation.
   - Adjust stroke width if needed and click "Submit" to process.
3. **View Results**:
   - The recognized equation will be displayed in LaTeX format (e premiers.g., `2 + 3` as `2 + 3`).
   - For simple expressions, the solved result (e.g., `2 + 3 = 5`) will be shown.

### Example
- **Input**: Image or canvas drawing of "2 + 3 ="
- **Output**: 
  - LaTeX: `2 + 3`
  - Result: `5`

## Dataset
The model is trained on the [Kaggle Handwritten Math Symbols Dataset](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols), which includes 45x45 pixel PNG images of digits, operators, and variables. Additional data augmentation techniques (e.g., rotation, scaling) are applied to improve model robustness.

## Project Structure
```
image-math-hand-written-recong/
├── data/
│   ├── train/
│   ├── test/
│   └── examples/
├── models/
│   └── cnn_model.h5
├── src/
│   ├── preprocess.py
│   ├── recognize.py
│   └── python app_new1.py

├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
├── requirements.txt
├── README.md
└── LICENSE
```

- `data/`: Contains training, testing, and example datasets.
- `models/`: Stores the trained CNN model (`cnn_model.h5`).
- `src/`: Source code for image preprocessing, symbol recognition, and the Flask web app.
- `static/`: Static files (CSS, JavaScript, images) for the web interface.
- `templates/`: HTML templates for the Flask frontend.

## Model Details
- **Architecture**: A Convolutional Neural Network (CNN) based on LeNet for symbol classification, with a post-processing step to interpret symbol sequences.
- **Preprocessing**:
  - **Image Processing**: Grayscale conversion, binarization, and noise removal using OpenCV.
  - **Segmentation**: Contour detection to isolate individual symbols, resized to 45x45 pixels.
- **Training**:
  - Trained on 80% of the Kaggle dataset, with 20% for validation.
  - Achieves ~95% accuracy on digit and operator recognition (based on validation set).
- **Output**: Recognized symbols are converted to LaTeX using a rule-based parser or evaluated using Python’s `eval()` for simple expressions.

## Dependencies
The project relies on the following Python libraries:
- **OpenCV**: For image preprocessing and segmentation.
- **TensorFlow**: For building and training the CNN model.
- **Flask**: For the web application backend.
- **NumPy**: For numerical computations.
- **Pillow**: For image handling.
See `requirements.txt` for the full list.

## Limitations
- **Symbol Support**: Limited to digits (0-9), basic operators (+, -, *, /, =), and variables (x, y). Advanced symbols (e.g., integrals, fractions) are not supported.
- **Image Constraints**: Images must be clear, with black symbols on a white background and minimal noise.
- **Development Stage**: The `develop` branch may include experimental features or incomplete functionality.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please report issues or bugs via the [GitHub Issues](https://github.com/Basavask/image-math-hand-written-recong/issues) page.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Technologies Used
The following technologies were used in this project:
- **Python**: General-purpose programming language for backend and model development.  
  [Website](https://www.python.org/)
- **OpenCV**: Library for computer vision and image processing.  
  [Website](https://opencv.org/)
- **TensorFlow**: Deep learning framework for building and training the CNN model.  
  [Website](https://www.tensorflow.org/)
- **Flask**: Lightweight web framework for the application’s backend.  
  [Website](https://flask.palletsprojects.com/)
- **NumPy**: Library for numerical computations and array operations.  
  [Website](https://numpy.org/)
- **Pillow**: Python Imaging Library for handling image files.  
  [Website](https://python-pillow.org/)
