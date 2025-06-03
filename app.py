from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import cv2
import joblib
import pickle
import warnings
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
import logging
import sys
from model_utils import load_models_and_assets, preprocess_single_image

# Add the parent directory to the path to allow importing from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Ganti dengan secret key yang aman

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model Configuration ---
MODEL_PATH_PKL = 'models/'
os.makedirs(MODEL_PATH_PKL, exist_ok=True)

# Initialize global variables
models = {}
model_loading_success = False

# Function to load models
def load_application_models():
    global models, model_loading_success
    
    logger.info(f"🔍 Checking model directory: {os.path.abspath(MODEL_PATH_PKL)}")
    
    if not os.path.exists(MODEL_PATH_PKL):
        logger.error(f"❌ Model directory not found: {os.path.abspath(MODEL_PATH_PKL)}")
        return False
    
    logger.info(f"✅ Model directory found: {os.path.abspath(MODEL_PATH_PKL)}")
    logger.info(f"📂 Directory contents: {os.listdir(MODEL_PATH_PKL)}")
    
    try:
        logger.info("🔍 Attempting to load models and assets...")
        loaded_assets = load_models_and_assets(MODEL_PATH_PKL)
        
        if not loaded_assets:
            logger.error("❌ Failed to load models and assets - load_models_and_assets returned None")
            return False
            
        # Store all components in the models dictionary
        models = {
            'models': loaded_assets.get('models', {}),
            'scaler': loaded_assets.get('scaler'),
            'label_encoder': loaded_assets.get('label_encoder'),
            'preprocessor': loaded_assets.get('preprocessor')
        }
        
        # Log loaded components
        logger.info(f"📦 Loaded models: {list(models['models'].keys())}")
        logger.info(f"📦 Scaler loaded: {'✅' if models['scaler'] is not None else '❌'}")
        logger.info(f"📦 Label Encoder loaded: {'✅' if models['label_encoder'] is not None else '❌'}")
        logger.info(f"📦 Preprocessor loaded: {'✅' if models['preprocessor'] is not None else '❌'}")
        
        # Verify we have at least one classifier model
        if models['models']:
            logger.info(f"✅ Found classifier models: {list(models['models'].keys())}")
            model_loading_success = True
            logger.info("✅ Successfully loaded all required models and components")
            return True
        else:
            logger.error("❌ No classifier models were loaded successfully")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}", exc_info=True)
        return False

# Load models when the application starts
model_loading_success = load_application_models()

# Log final status
if model_loading_success:
    logger.info("🎉 Model loading completed successfully!")
else:
    logger.error("❌ Model loading failed. The application may not function correctly.")
    # Initialize empty models to prevent KeyErrors
    models = {
        'models': {},
        'scaler': None, 
        'label_encoder': None, 
        'preprocessor': None
    }

# Model paths
MODEL_PATHS = {
    'svm': os.path.join(MODEL_PATH_PKL, 'svm_model.pkl'),
    'knn': os.path.join(MODEL_PATH_PKL, 'knn_model.pkl'),
    'rf': os.path.join(MODEL_PATH_PKL, 'random_forest_model.pkl'),
    'label_encoder': os.path.join(MODEL_PATH_PKL, 'label_encoder.pkl'),
    'scaler': os.path.join(MODEL_PATH_PKL, 'scaler.pkl')
}

# --- Image Processing Configuration ---
TARGET_IMAGE_SIZE = (128, 128)
BLUR_KERNEL = (3, 3)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_array_rgb):
    """
    Preprocessing gambar:
    1. Gaussian Blur untuk mengurangi noise
    2. Konversi ke HSV color space
    3. Apply CLAHE pada channel V untuk meningkatkan kontras
    """
    try:
        # 1. Gaussian Blur
        blurred = cv2.GaussianBlur(img_array_rgb, BLUR_KERNEL, 0)
        
        # 2. Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        
        # 3. Apply CLAHE to V channel
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        
        return hsv
    except Exception as e:
        logger.error(f"Error dalam preprocessing: {e}")
        raise

@app.route('/')
def index():
    try:
        # Check if models are loaded
        if not models or 'models' not in models:
            logger.error("No models loaded when accessing index page")
            return render_template('error.html', 
                                message='Models are not loaded. Please check the server logs for details.')
        
        # Get list of available models
        model_mapping = {
            'Svm': 'svm',
            'Knn': 'knn',
            'Random Forest': 'rf'
        }
        
        available_models = []
        for model_name, model_key in model_mapping.items():
            if model_name in models['models'] and models['models'][model_name] is not None:
                available_models.append({
                    'value': model_key,
                    'name': model_name.upper() if model_name != 'Random Forest' else 'Random Forest'
                })
        
        # If no models are available, show an error message
        if not available_models:
            logger.error("No valid models found in the loaded models")
            return render_template('error.html', 
                                message='No valid models found. Please check the server logs for details.')
        
        # Check if required components are available
        missing_components = []
        if not models.get('scaler'):
            missing_components.append('scaler')
        if not models.get('label_encoder'):
            missing_components.append('label_encoder')
        if not models.get('preprocessor'):
            missing_components.append('preprocessor')
            
        if missing_components:
            logger.warning(f"Missing required components: {', '.join(missing_components)}")
        
        # Prepare component status for the template
        components_status = {
            'scaler': models.get('scaler') is not None,
            'label_encoder': models.get('label_encoder') is not None,
            'preprocessor': models.get('preprocessor') is not None
        }
        
        # Render the main page with available models and component status
        return render_template('index.html', 
                            models=available_models,
                            default_model=available_models[0]['value'],
                            components_status=components_status)
    
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('error.html',
                            message='An error occurred while loading the page. Please check the server logs for details.')

@app.route('/predict_cover', methods=['POST'])
def predict_cover():
    """Endpoint untuk prediksi kategori sampul buku"""
    
    # Check if models are loaded
    if not models:
        return jsonify({
            'error': 'Model tidak dimuat. Silakan muat ulang halaman atau coba lagi nanti.',
            'missing_models': ['all']
        }), 500
    
    # Validasi model essential (minimal yang dibutuhkan untuk prediksi)
    essential_models = ['label_encoder', 'scaler']
    missing_essential = [model for model in essential_models if model not in models or models[model] is None]
    
    if missing_essential:
        return jsonify({
            'error': f'Model essential tidak dimuat: {", ".join(missing_essential)}. Aplikasi tidak dapat berjalan.',
            'missing_models': missing_essential
        }), 500
        
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request', 'status': 'error'}), 400
        
    file = request.files['image']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
        
    if not file:
        return jsonify({'error': 'No file uploaded', 'status': 'error'}), 400
        
    # Map frontend model names to backend model names
    model_mapping = {
        'svm': 'Svm',
        'knn': 'Knn',
        'rf': 'Random Forest'
    }
    
    # Get the model type from the form data
    model_type = request.form.get('model_type', 'svm')
    model_name = model_mapping.get(model_type, 'Svm')  # Default to SVM if not found
    
    # Check if the selected model is available
    if 'models' not in models or model_name not in models['models'] or models['models'][model_name] is None:
        available_models = [k.lower() for k, v in models.get('models', {}).items() 
                          if v is not None and k in model_mapping.values()]
        return jsonify({
            'error': f'Model {model_name} not available',
            'available_models': available_models,
            'status': 'error'
        }), 400

    img_path = None
    try:
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        logger.info(f"Temporary image saved to {img_path}")

        # Get the preprocessor
        preprocessor = models.get('preprocessor')
        if preprocessor is None:
            return jsonify({'error': 'Image preprocessor not available', 'status': 'error'}), 500

        # Preprocess the image
        logger.info("Preprocessing image...")
        features = preprocess_single_image(img_path, preprocessor)
        if features is None:
            return jsonify({'error': 'Failed to process image', 'status': 'error'}), 400

        # Scale the features
        scaler = models.get('scaler')
        if scaler is None:
            return jsonify({'error': 'Feature scaler not available', 'status': 'error'}), 500
            
        features_scaled = scaler.transform([features])
        
        # Make prediction
        model = models['models'][model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)[0].tolist()
            # Convert numpy types to native Python types for JSON serialization
            probabilities = [float(p) for p in probabilities]
            logger.info(f"Got prediction probabilities: {probabilities}")
        except (AttributeError, Exception) as e:
            logger.warning(f"Could not get probabilities: {str(e)}")
            probabilities = None
        
        # Get the predicted label
        label_encoder = models.get('label_encoder')
        if label_encoder is not None:
            try:
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                logger.info(f"Decoded label: {predicted_label}")
            except Exception as e:
                logger.warning(f"Error in label decoding: {str(e)}")
                predicted_label = str(prediction)
        else:
            predicted_label = str(prediction)
        
        # Map model name for display
        display_model_names = {
            'Svm': 'SVM',
            'Knn': 'KNN',
            'Random Forest': 'Random Forest'
        }
        
        response_data = {
            'prediction': predicted_label,
            'probabilities': probabilities,
            'model_used': display_model_names.get(model_name, model_name),
            'status': 'success'
        }
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e),
            'status': 'error'
        }), 500
    
    finally:
        # Clean up the temporary file
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
                logger.info("Temporary file deleted successfully")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {img_path}: {str(e)}")

@app.route('/model_status', methods=['GET'])
def model_status():
    """Endpoint to check the status of models and components"""
    if not models:
        return jsonify({
            'status': 'error',
            'message': 'No models loaded',
            'available_models': []
        }), 500
    
    # Check which models are available
    model_status = {}
    available_models = []
    
    # Check each model file
    for model_key, model_path in MODEL_PATHS.items():
        try:
            file_exists = os.path.exists(model_path)
            model_loaded = False
            
            # Check if model is loaded in memory
            if 'models' in models and model_key in models['models']:
                model_loaded = models['models'][model_key] is not None
            
            model_status[model_key] = {
                'file_exists': file_exists,
                'loaded': model_loaded,
                'path': model_path
            }
            
            if model_loaded:
                available_models.append(model_key.upper())
                
        except Exception as e:
            logger.error(f"Error checking status for {model_key}: {str(e)}")
            model_status[model_key] = {
                'file_exists': False,
                'loaded': False,
                'error': str(e)
            }
    
    # Check required components
    components_status = {
        'scaler': models.get('scaler') is not None,
        'label_encoder': models.get('label_encoder') is not None,
        'preprocessor': models.get('preprocessor') is not None
    }
    
    all_components_loaded = all(components_status.values())
    
    return jsonify({
        'status': 'success',
        'models': model_status,
        'components': components_status,
        'available_models': available_models,
        'all_components_loaded': all_components_loaded,
        'ready_for_prediction': len(available_models) > 0 and all_components_loaded
    })

@app.route('/debug_features', methods=['POST'])
def debug_features():
    """Endpoint to debug feature extraction without prediction"""
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'error': 'No image file uploaded'
        }), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'error': 'No file selected'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'error': 'File format not supported. Use PNG, JPG, or JPEG'
        }), 400

    img_path = None
    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        logger.info(f"Temporary image saved to {img_path}")

        # Get the preprocessor
        preprocessor = models.get('preprocessor')
        if preprocessor is None:
            return jsonify({
                'status': 'error',
                'error': 'Image preprocessor not available'
            }), 500

        # Preprocess the image
        logger.info("Preprocessing image for feature extraction...")
        features = preprocess_single_image(img_path, preprocessor)
        
        if features is None:
            return jsonify({
                'status': 'error',
                'error': 'Failed to process image'
            }), 400
        
        # Get additional feature information if available
        feature_info = {
            'shape': features.shape if hasattr(features, 'shape') else None,
            'dtype': str(features.dtype) if hasattr(features, 'dtype') else str(type(features)),
            'min': float(features.min()) if hasattr(features, 'min') else None,
            'max': float(features.max()) if hasattr(features, 'max') else None,
            'mean': float(features.mean()) if hasattr(features, 'mean') else None,
            'sample': features.flatten()[:10].tolist() if hasattr(features, 'flatten') else None
        }
        
        return jsonify({
            'status': 'success',
            'features': feature_info,
            'message': 'Feature extraction completed successfully',
            'image_size': TARGET_IMAGE_SIZE,
            'feature_breakdown': {
                'info': 'Check logs for detailed breakdown'
            }
        })

    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'An error occurred during feature extraction',
            'details': str(e)
        }), 500
    
    finally:
        # Clean up the temporary file
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
                logger.info("Temporary file deleted successfully")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {img_path}: {str(e)}")

@app.route('/available_models', methods=['GET'])
def available_models():
    """Endpoint to get the list of available models for prediction"""
    if not models or 'models' not in models:
        return jsonify({
            'status': 'error',
            'message': 'No models loaded',
            'available_models': []
        }), 500
    
    available = []
    model_mapping = {
        'Svm': 'svm',
        'Knn': 'knn',
        'Random Forest': 'rf'
    }
    
    # Check each model type
    for model_name, model_key in model_mapping.items():
        if model_name in models['models'] and models['models'][model_name] is not None:
            available.append({
                'id': model_key,
                'name': model_name,
                'display_name': model_name.upper() if model_name != 'Random Forest' else 'Random Forest'
            })
    
    return jsonify({
        'status': 'success',
        'available_models': available,
        'default_model': available[0]['id'] if available else None
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size too large error"""
    return jsonify({'error': 'File terlalu besar. Maksimal 5MB'}), 413

if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5001)