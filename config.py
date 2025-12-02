"""
Configuration file for Anomaly Detection in Patient Vital Signs
Contains all hyperparameters, paths, and constants
"""

import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, EDA_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DATASET_PATH = BASE_DIR / "iot_health_monitoring_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.pkl"

# ==================== FEATURE DEFINITIONS ====================
FEATURE_COLUMNS = [
    'heart_rate',
    'blood_oxygen',
    'blood_pressure_systolic',
    'blood_pressure_diastolic',
    'glucose_level',
    'body_temperature',
    'respiratory_rate',
    'activity_level',
    'sleep_quality',
    'stress_level',
    'hrv_sdnn',
    'steps_count',
    'calories_burned'
]

# Normal ranges for vital signs (for validation and context)
# Normal ranges for vital signs (Medical Constraints)
NORMAL_RANGES = {
    'heart_rate': (30, 197),
    'blood_oxygen': (85.8, 100),
    'blood_pressure_systolic': (76, 198),
    'blood_pressure_diastolic': (51, 137),
    'glucose_level': (32.5, 165.7),
    'body_temperature': (96.42, 101.12),
    'respiratory_rate': (8.4, 35),
    'activity_level': (0, 1),
    'sleep_quality': (0, 1),
    'stress_level': (0, 1),
    'hrv_sdnn': (5, 109.1),
    'steps_count': (519, 11998),
    'calories_burned': (1000, 2799)
}

# Columns to exclude from training
EXCLUDE_COLUMNS = ['timestamp', 'device_id', 'patient_id', 'health_event']

# ==================== SEVERITY THRESHOLDS ====================
# Based on Anomaly Score (0.0 - 1.0)
SEVERITY_THRESHOLDS = {
    'Normal': (0.00, 0.30),
    'Borderline Anomaly': (0.31, 0.60),
    'Highly Abnormal Anomaly': (0.61, 0.85),
    'Extreme Abnormal Anomaly': (0.86, 1.00)
}

SEVERITY_MAP = {
    'Normal': 0,
    'Borderline Anomaly': 1,
    'Highly Abnormal Anomaly': 2,
    'Extreme Abnormal Anomaly': 3
}

# ==================== HEALTH EVENT PATTERNS ====================
# Reference patterns for medical explanation matching
HEALTH_EVENT_PATTERNS = {
    'Normal': {
        'heart_rate': 75, 'blood_oxygen': 97, 'blood_pressure_systolic': 120, 
        'glucose_level': 100, 'stress_level': 0.1, 'hrv_sdnn': 50
    },
    'Cardiac Arrhythmia': {
        'heart_rate': 92, 'blood_oxygen': 93, 'stress_level': 0.78, 'hrv_sdnn': 19
    },
    'Hypoglycemia': {
        'glucose_level': 55, 'blood_pressure_systolic': 110, 'stress_level': 0.67
    },
    'Hypertensive Crisis': {
        'blood_pressure_systolic': 164, 'blood_pressure_diastolic': 105, 
        'heart_rate': 105, 'stress_level': 0.85
    }
}

# ==================== DATA PREPROCESSING ====================
SCALER_TYPE = 'standard'  # 'standard' or 'minmax'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== ISOLATION FOREST PARAMETERS ====================
ISOLATION_FOREST_DEFAULT = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 0.15,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Hyperparameter tuning ranges for Isolation Forest
ISOLATION_FOREST_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.75, 1.0],
    'contamination': [0.05, 0.1, 0.15, 0.2]
}

# ==================== AUTOENCODER PARAMETERS ====================
AUTOENCODER_DEFAULT = {
    'encoding_dims': [64, 32, 16],  # Encoder layer sizes
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout_rate': 0.2,
    'patience': 10  # Early stopping patience
}

# Hyperparameter tuning ranges for Autoencoder (Optuna)
AUTOENCODER_OPTUNA_PARAMS = {
    'hidden_dim_1': (32, 128),
    'hidden_dim_2': (16, 64),
    'hidden_dim_3': (8, 32),
    'learning_rate': (1e-4, 1e-2),
    'batch_size': [16, 32, 64],
    'dropout_rate': (0.1, 0.5),
    'n_trials': 50
}

# ==================== LSTM AUTOENCODER PARAMETERS ====================
LSTM_AUTOENCODER_DEFAULT = {
    'lstm_units': 64,
    'num_layers': 2,
    'sequence_length': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout_rate': 0.2,
    'patience': 10
}

# Hyperparameter tuning ranges for LSTM Autoencoder (Optuna)
LSTM_OPTUNA_PARAMS = {
    'lstm_units': (32, 128),
    'num_layers': [1, 2, 3],
    'sequence_length': [5, 10, 20, 30],
    'learning_rate': (1e-4, 1e-2),
    'batch_size': [16, 32, 64],
    'dropout_rate': (0.1, 0.5),
    'n_trials': 50
}

# ==================== TRAINING PARAMETERS ====================
DEVICE = 'cuda'  # Will auto-detect in code
GRADIENT_CLIP_VALUE = 1.0
VALIDATION_SPLIT = 0.2

# ==================== VISUALIZATION PARAMETERS ====================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (14, 6)
DPI = 100

# Color scheme for severity levels
SEVERITY_COLORS = {
    'normal': '#28a745',      # Green
    'mild': '#ffc107',        # Yellow
    'moderate': '#fd7e14',    # Orange
    'severe': '#dc3545'       # Red
}

# ==================== FLASK APP PARAMETERS ====================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# ==================== SHAP PARAMETERS ====================
SHAP_SAMPLES = 100  # Number of samples for SHAP background
TOP_FEATURES = 5    # Top N features to show in explanations

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
