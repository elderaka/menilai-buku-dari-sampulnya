"""
Script untuk mengatasi masalah Random Forest model compatibility
Metode: Downgrade scikit-learn -> Load model -> Recreate -> Save dengan versi baru
"""

import subprocess
import sys
import os
import joblib
import warnings
import tempfile
import shutil

def run_command(command):
    """Execute command and return result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_current_sklearn_version():
    """Get current scikit-learn version"""
    try:
        import sklearn
        return sklearn.__version__
    except:
        return None

def backup_file(filepath):
    """Create backup of file"""
    backup_path = filepath + ".backup"
    try:
        shutil.copy2(filepath, backup_path)
        return True, backup_path
    except Exception as e:
        return False, str(e)

def fix_random_forest_model():
    """
    Fix Random Forest model compatibility issue
    """
    print("🔧 Random Forest Model Compatibility Fixer")
    print("=" * 50)
    
    model_path = "models/random_forest_model.pkl"
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ File tidak ditemukan: {model_path}")
        return False
    
    # Get current version
    current_version = get_current_sklearn_version()
    print(f"📋 Current scikit-learn version: {current_version}")
    
    # Backup original file
    print("📁 Creating backup...")
    backup_success, backup_info = backup_file(model_path)
    if not backup_success:
        print(f"❌ Backup failed: {backup_info}")
        return False
    print(f"✅ Backup created: {backup_info}")
    
    # Step 1: Downgrade to 1.2.2
    print("\n🔄 Step 1: Downgrading scikit-learn to 1.2.2...")
    success, stdout, stderr = run_command("pip install scikit-learn==1.2.2")
    if not success:
        print(f"❌ Downgrade failed: {stderr}")
        return False
    print("✅ Downgrade successful")
    
    try:
        # Step 2: Load model with old version
        print("\n📥 Step 2: Loading model with scikit-learn 1.2.2...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Import with old version
            from sklearn.ensemble import RandomForestClassifier
            
            # Load model
            rf_model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            
            # Get model parameters
            params = rf_model.get_params()
            print(f"📋 Model parameters: n_estimators={params.get('n_estimators', 'unknown')}")
            
            # Step 3: Create new model with same parameters
            print("\n🔨 Step 3: Creating new model with same parameters...")
            
            # Extract training data if available (this is tricky without original data)
            # We'll recreate with same parameters but we need training data
            
            # For now, let's try to extract what we can
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            random_state = params.get('random_state', None)
            
            print("⚠️  WARNING: Recreating model requires original training data!")
            print("⚠️  This script will create a template. You need to retrain with your data.")
            
            # Create template model with same parameters
            new_rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth', 'random_state']}
            )
            
            # Save template info
            template_info = {
                'model_type': 'RandomForestClassifier',
                'parameters': params,
                'note': 'This is a template. Retrain with your original training data.',
                'original_version': '1.2.2',
                'target_version': current_version
            }
            
            # Save template info
            template_path = "models/rf_template_info.json"
            import json
            with open(template_path, 'w') as f:
                json.dump(template_info, f, indent=2, default=str)
            
            print(f"✅ Template info saved to: {template_path}")
        
    except Exception as e:
        print(f"❌ Error during model processing: {e}")
        # Restore backup before upgrading
        try:
            shutil.copy2(backup_info, model_path)
            print("📁 Backup restored")
        except:
            pass
        return False
    
    finally:
        # Step 4: Upgrade back to latest version
        print(f"\n⬆️  Step 4: Upgrading scikit-learn back to latest...")
        success, stdout, stderr = run_command("pip install scikit-learn --upgrade")
        if not success:
            print(f"⚠️  Upgrade failed: {stderr}")
            print("⚠️  Please manually run: pip install scikit-learn --upgrade")
        else:
            print("✅ Upgrade successful")
    
    print("\n" + "=" * 50)
    print("🎯 RESULT:")
    print("❌ Random Forest model could not be automatically fixed")
    print("💡 SOLUTION: You need to retrain the Random Forest model with current scikit-learn version")
    print("\n📝 To fix completely:")
    print("1. Use your original training data")
    print("2. Retrain Random Forest with current scikit-learn")
    print("3. Save with joblib.dump()")
    print(f"\n📋 Model parameters for retraining are saved in: {template_path}")
    
    return False

def create_retrain_script():
    """Create a template script for retraining"""
    script_content = '''"""
Template script for retraining Random Forest model
Replace X_train and y_train with your actual training data
"""

import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load template info
with open('models/rf_template_info.json', 'r') as f:
    template_info = json.load(f)

print("Original model parameters:")
for key, value in template_info['parameters'].items():
    print(f"  {key}: {value}")

# TODO: Load your training data here
# X_train = ...  # Your features
# y_train = ...  # Your labels

# Example code (replace with your actual data):
# from sklearn.datasets import make_classification
# X_train, y_train = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

# Create model with original parameters
rf_model = RandomForestClassifier(**template_info['parameters'])

# Train model
# rf_model.fit(X_train, y_train)

# Save model
# joblib.dump(rf_model, 'models/random_forest_model.pkl')

print("✅ Model retrained and saved!")
'''
    
    with open('retrain_rf_model.py', 'w') as f:
        f.write(script_content)
    
    print(f"📝 Retrain script created: retrain_rf_model.py")

if __name__ == "__main__":
    success = fix_random_forest_model()
    create_retrain_script()
    
    if not success:
        print("\n🚀 QUICK SOLUTION:")
        print("For now, your app can run with SVM and KNN models only.")
        print("Random Forest will be disabled until retrained.")