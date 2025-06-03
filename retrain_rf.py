"""
Script to retrain the Random Forest model with current scikit-learn version.
This ensures compatibility with the current system.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

# Paths
MODEL_DIR = 'models'
RF_OLD_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
RF_NEW_PATH = os.path.join(MODEL_DIR, 'random_forest_model_new.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

def load_training_data():
    """
    Load preprocessed training data.
    If you have a saved dataset, modify this to load it.
    """
    logger.info("Looking for existing feature files...")
    
    try:
        # Try to load preprocessed features if they exist
        features_train_path = 'features_train.npy'
        features_test_path = 'features_test.npy'
        labels_train_path = 'labels_train.npy'
        labels_test_path = 'labels_test.npy'
        
        if os.path.exists(features_train_path) and os.path.exists(labels_train_path):
            logger.info("Loading saved feature files...")
            X_train = np.load(features_train_path)
            y_train = np.load(labels_train_path)
            logger.info(f"Loaded training data with shape {X_train.shape}")
            
            # Load test data if available (for evaluation)
            if os.path.exists(features_test_path) and os.path.exists(labels_test_path):
                X_test = np.load(features_test_path)
                y_test = np.load(labels_test_path)
                logger.info(f"Loaded test data with shape {X_test.shape}")
            else:
                X_test, y_test = None, None
                logger.warning("Test data not found, will train without evaluation")
            
            return X_train, y_train, X_test, y_test
        else:
            logger.error("Feature files not found!")
            raise FileNotFoundError("Feature files not found. You need to either run feature extraction first or provide a path to your features.")
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_random_forest(X_train, y_train, scaler):
    """
    Train a Random Forest model with the same parameters as the original.
    """
    logger.info("Starting Random Forest model training...")
    
    # Scale the features
    X_train_scaled = scaler.transform(X_train.astype(np.float32))
    logger.info(f"Scaled training data shape: {X_train_scaled.shape}")
    
    # Create a Random Forest model with the same parameters as in your notebook
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    logger.info("Training Random Forest model...")
    rf_model.fit(X_train_scaled, y_train)
    logger.info("Random Forest model training completed")
    
    return rf_model

def save_model(model, filepath):
    """
    Save the model to disk.
    """
    try:
        logger.info(f"Saving model to {filepath}...")
        joblib.dump(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def main():
    """
    Main function to retrain and save the Random Forest model.
    """
    logger.info("Starting Random Forest model retraining process")
    
    # Check if required files exist
    if not os.path.exists(SCALER_PATH):
        logger.error(f"Scaler not found at {SCALER_PATH}")
        return
    
    if not os.path.exists(LABEL_ENCODER_PATH):
        logger.error(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        return
    
    # Load scaler
    try:
        logger.info(f"Loading scaler from {SCALER_PATH}...")
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return
    
    # Load label encoder
    try:
        logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}...")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("Label encoder loaded successfully")
    except Exception as e:
        logger.error(f"Error loading label encoder: {e}")
        return
    
    try:
        # Load training data
        X_train, y_train, X_test, y_test = load_training_data()
        
        # Train the Random Forest model
        rf_model = train_random_forest(X_train, y_train, scaler)
        
        # Save the model
        if save_model(rf_model, RF_NEW_PATH):
            # If successful, replace the old model (optional)
            if os.path.exists(RF_OLD_PATH):
                logger.info(f"Backing up original model to {RF_OLD_PATH}.bak")
                os.rename(RF_OLD_PATH, f"{RF_OLD_PATH}.bak")
            
            logger.info(f"Renaming new model to replace the original")
            os.rename(RF_NEW_PATH, RF_OLD_PATH)
            logger.info("Random Forest model has been successfully retrained and replaced")
        
        # Evaluate if test data is available
        if X_test is not None and y_test is not None:
            X_test_scaled = scaler.transform(X_test.astype(np.float32))
            accuracy = rf_model.score(X_test_scaled, y_test)
            logger.info(f"New model test accuracy: {accuracy:.4f}")
    
    except Exception as e:
        logger.error(f"Error in retraining process: {e}")

if __name__ == "__main__":
    main()
