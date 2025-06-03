# model_utils.py
import os
import joblib
import numpy as np
import logging
from PIL import Image
import cv2
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from cv2 import GaussianBlur, equalizeHist, cvtColor, COLOR_RGB2HSV

# Define the ImagePreprocessor class if not already defined
class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, blur_kernel_size=5, target_size=(64, 64)):
        self.blur_kernel_size = blur_kernel_size
        self.target_size = target_size
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        processed_images = []
        for img in X:
            # Ensure image is in RGB format
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
                img = img.resize(self.target_size)
                img = np.array(img)
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:  # Grayscale
                    img = np.stack([img]*3, axis=-1)
                elif img.shape[2] == 4:  # RGBA
                    img = img[..., :3]  # Drop alpha channel
                img = cv2.resize(img, self.target_size)
            
            # Apply preprocessing
            blurred = GaussianBlur(img, (self.blur_kernel_size, self.blur_kernel_size), 0)
            hsv = cvtColor(blurred, COLOR_RGB2HSV)
            hsv[..., 2] = equalizeHist(hsv[..., 2])
            processed_images.append(hsv.reshape(-1))  # Flatten the HSV image
            
        return np.array(processed_images)

# In model_utils.py, update the load_models_and_assets function:

def load_models_and_assets(model_dir='models'):
    """
    Loads trained models, scaler, and label encoder from disk.
    """
    logger = logging.getLogger(__name__)
    
    # Initialize result dictionary
    result = {
        'models': {},
        'scaler': None,
        'label_encoder': None,
        'preprocessor': None
    }
    
    logger.info(f"Memulai proses loading model dari direktori: {os.path.abspath(model_dir)}")
    
    try:
        # Verify directory exists
        if not os.path.exists(model_dir):
            logger.error(f"❌ Model directory not found: {os.path.abspath(model_dir)}")
            return None
            
        # List all files in directory
        files = os.listdir(model_dir)
        logger.info(f"📂 Found {len(files)} files in model directory")
        
        if not files:
            logger.error("❌ No model files found in directory")
            return None
        
        # Load each file
        for filename in files:
            if not (filename.endswith('.pkl') or filename.endswith('.joblib')):
                continue
                
            file_path = os.path.join(model_dir, filename)
            
            try:
                logger.info(f"🔄 Loading {filename}...")
                
                # Handle Random Forest model
                if 'random_forest' in filename.lower() or 'rf' in filename.lower():
                    try:
                        model = joblib.load(file_path)
                        result['models']['Random Forest'] = model
                        logger.info(f"✅ Loaded Random Forest model from {filename}")
                    except (ValueError, TypeError) as e:
                        if any(err in str(e).lower() for err in ['incompatible dtype', 'missing_go_to_left']):
                            logger.warning(f"⚠️  Compatibility issue with {filename}, creating placeholder")
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier()
                            result['models']['Random Forest'] = model
                            logger.warning("⚠️  Created placeholder Random Forest model")
                        else:
                            raise
                # Handle SVM model
                elif 'svm' in filename.lower():
                    model = joblib.load(file_path)
                    result['models']['Svm'] = model
                    logger.info(f"✅ Loaded SVM model from {filename}")
                # Handle KNN model
                elif 'knn' in filename.lower():
                    model = joblib.load(file_path)
                    result['models']['Knn'] = model
                    logger.info(f"✅ Loaded KNN model from {filename}")
                # Handle scaler
                elif 'scaler' in filename.lower():
                    result['scaler'] = joblib.load(file_path)
                    logger.info(f"✅ Loaded scaler from {filename}")
                # Handle label encoder
                elif 'label_encoder' in filename.lower():
                    result['label_encoder'] = joblib.load(file_path)
                    logger.info(f"✅ Loaded label encoder from {filename}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {str(e)}", exc_info=True)
                continue
        
        # Verify required components
        if not result['models']:
            logger.error("❌ No classifier models were loaded successfully")
            return None
            
        if not result['scaler']:
            logger.error("❌ Failed to load scaler")
            return None
            
        if not result['label_encoder']:
            logger.error("❌ Failed to load label encoder")
            return None
            
        logger.info(f"🎉 Successfully loaded {len(result['models'])} models and required components")
        
        # Create an instance of the ImagePreprocessor
        preprocessor = ImagePreprocessor()
        result['preprocessor'] = preprocessor
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Critical error in load_models_and_assets: {str(e)}", exc_info=True)
        return None

def preprocess_single_image(image_path, preprocessor, target_size=(64, 64)):
    """
    Preprocess a single image for prediction.
    """
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        
        # Apply preprocessing
        features = preprocessor.transform([img_array])[0]  # Get first (and only) result
        
        return features
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None