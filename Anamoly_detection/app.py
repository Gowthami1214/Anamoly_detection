"""
Flask Web Application for Real-Time Anomaly Detection
Provides REST API and web interface for patient vital signs anomaly detection
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import config
from data_preprocessing import get_scaler
from models.isolation_forest_model import IsolationForestDetector
from models.autoencoder_model import AutoencoderDetector
from models.lstm_autoencoder_model import LSTMAutoencoderDetector

app = Flask(__name__)

# Add route to serve static files from outputs directory
@app.route('/static/outputs/<path:filename>')
def serve_outputs(filename):
    """Serve files from outputs directory"""
    return send_from_directory(config.OUTPUTS_DIR, filename)

# Global variables for loaded model
loaded_model = None
loaded_scaler = None
model_config = None
model_type = None

def load_best_model():
    """Load the best trained model and scaler"""
    global loaded_model, loaded_scaler, model_config, model_type
    
    print("Loading best model...")
    
    # Load model configuration
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model configuration not found at {config.BEST_MODEL_PATH}. "
            "Please run train_all_models.py first."
        )
    
    model_config = joblib.load(config.BEST_MODEL_PATH)
    model_path = model_config['model_path']
    model_name = model_config['model_name']
    
    print(f"Loading {model_name}...")
    
    # Load appropriate model
    if model_name == "Isolation Forest":
        loaded_model = IsolationForestDetector.load_model(model_path)
        model_type = "isolation_forest"
    elif model_name == "Autoencoder":
        loaded_model = AutoencoderDetector.load_model(model_path)
        model_type = "autoencoder"
    elif model_name == "LSTM Autoencoder":
        loaded_model = LSTMAutoencoderDetector.load_model(model_path)
        model_type = "lstm_autoencoder"
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load scaler
    loaded_scaler = get_scaler(model_config['scaler_path'])
    
    print(f"[OK] {model_name} loaded successfully!")
    print(f"[OK] Scaler loaded successfully!")
    
    return loaded_model, loaded_scaler, model_config


def validate_input(data):
    """Validate input vital signs data"""
    required_features = config.FEATURE_COLUMNS
    
    errors = []
    for feature in required_features:
        if feature not in data:
            errors.append(f"Missing required field: {feature}")
        else:
            try:
                value = float(data[feature])
                
                # Basic sanity check - must be non-negative for most features
                if value < 0 and feature not in ['stress_level', 'activity_level', 'sleep_quality']:
                    errors.append(f"{feature} cannot be negative: {value}")
                
                # Very lenient range check - only reject completely unrealistic values
                if feature in config.NORMAL_RANGES:
                    min_val, max_val = config.NORMAL_RANGES[feature]
                    # Allow 0 to 3x the max (to catch anomalies, not reject them)
                    if value < 0 or value > max_val * 3:
                        errors.append(f"{feature} value {value} is outside physiologically possible range (0 - {max_val * 3:.1f})")
                        
            except (ValueError, TypeError):
                errors.append(f"Invalid value for {feature}: {data[feature]}")
    
    return errors


def get_severity_color(severity):
    """Get color code for severity level"""
    # Handle new verbose labels (e.g., "Highly Abnormal Anomaly" -> "highly")
    # Map to config colors
    key = severity.lower().split()[0]
    
    # Map new keys to old color keys if needed
    if key == 'highly': key = 'moderate'
    if key == 'extreme': key = 'severe'
    if key == 'borderline': key = 'mild'
    
    return config.SEVERITY_COLORS.get(key, '#6c757d')


def get_severity_description(severity):
    """Get human-readable description for severity level"""
    descriptions = {
        'normal': 'All vital signs are within normal ranges. No immediate concerns detected.',
        'mild': 'Minor deviations detected. Monitor patient and consider follow-up if symptoms persist.',
        'moderate': 'Significant anomalies detected. Medical attention recommended.',
        'severe': 'Critical anomalies detected. Immediate medical intervention may be required.'
    }
    return descriptions.get(severity, 'Unknown severity level')


@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html', 
                         feature_columns=config.FEATURE_COLUMNS,
                         normal_ranges=config.NORMAL_RANGES)


def find_closest_health_pattern(data):
    """
    Find the closest health event pattern based on input data
    
    Args:
        data: Dictionary of input vital signs
        
    Returns:
        tuple: (pattern_name, similarity_score)
    """
    best_match = 'Normal'
    min_distance = float('inf')
    
    # Normalize input data for comparison (simple min-max based on ranges)
    normalized_input = {}
    for feature, value in data.items():
        if feature in config.NORMAL_RANGES:
            min_val, max_val = config.NORMAL_RANGES[feature]
            range_span = max_val - min_val
            if range_span > 0:
                normalized_input[feature] = (float(value) - min_val) / range_span
    
    for pattern_name, pattern_features in config.HEALTH_EVENT_PATTERNS.items():
        distance = 0
        count = 0
        
        for feature, target_value in pattern_features.items():
            if feature in normalized_input and feature in config.NORMAL_RANGES:
                # Normalize target value
                min_val, max_val = config.NORMAL_RANGES[feature]
                norm_target = (target_value - min_val) / (max_val - min_val)
                
                # Calculate squared difference
                distance += (normalized_input[feature] - norm_target) ** 2
                count += 1
        
        if count > 0:
            avg_distance = (distance / count) ** 0.5
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_match = pattern_name
                
    return best_match

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Prepare input data
        input_data = pd.DataFrame([{
            feature: float(data[feature]) 
            for feature in config.FEATURE_COLUMNS
        }])
        
        # Scale input
        input_scaled = loaded_scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=config.FEATURE_COLUMNS)
        
        # Get prediction
        if model_type == "isolation_forest":
            scores, labels, severity_info = loaded_model.predict(input_scaled_df)
            contributions = loaded_model.get_feature_contributions(input_scaled_df, top_n=config.TOP_FEATURES)
        elif model_type == "autoencoder":
            scores, labels, severity_info = loaded_model.predict(input_scaled_df)
            contributions = loaded_model.get_feature_contributions(input_scaled_df, top_n=config.TOP_FEATURES)
        elif model_type == "lstm_autoencoder":
            scores, labels, severity_info, _ = loaded_model.predict(input_scaled_df)
            contributions = []
        
        # Extract single values
        anomaly_score = float(scores[0])
        is_anomaly = int(labels[0])
        
        # Handle severity info
        if isinstance(severity_info, list):
            severity_label = severity_info[0]['severity_label']
            severity_class = severity_info[0]['severity_class']
        else:
            # Fallback
            severity_label = str(severity_info[0])
            severity_class = 0
            if severity_label != 'Normal':
                severity_class = 1
        
        # Format contributions
        top_features = []
        if isinstance(contributions, list) and len(contributions) > 0:
            for contrib in contributions:
                top_features.append({
                    'feature': contrib['feature'].replace('_', ' ').title(),
                    'contribution': float(contrib.get('contribution', 0)),
                    'value': float(contrib['value']),
                    'feature_name': contrib['feature']
                })
        
        # Find closest medical pattern (ONLY if severity_class >= 2)
        pattern_similarity = "None"
        closest_pattern = "Normal"
        
        if severity_class >= 2:
            closest_pattern = find_closest_health_pattern(data)
            if closest_pattern != 'Normal':
                pattern_similarity = closest_pattern
        
        # Generate dynamic explanation
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        explanation = ""
        
        if severity_class == 0:
            explanation = "No anomaly detected. All vitals within normal range."
        else:
            # Severity >= 1
            explanation = f"{severity_label} detected."
            
            # Add top contributors
            if top_features:
                top_details = []
                for f in top_features[:3]:
                    top_details.append(f"{f['feature']} ({f['value']:.1f})")
                explanation += f" Primary contributors: {', '.join(top_details)}."
            
            # Add pattern info only for significant anomalies (Severity >= 2)
            if severity_class >= 2 and pattern_similarity != "None":
                explanation += f" The vital signs pattern resembles: {pattern_similarity}."
        
        # Prepare response
        response = {
            'success': True,
            'anomaly_score': round(anomaly_score, 4),
            'severity_class': severity_class,
            'severity_label': severity_label,
            'is_anomaly': is_anomaly,
            'top_contributing_features': top_features,
            'explanation': explanation,
            'pattern_similarity': pattern_similarity,
            'timestamp': timestamp_str,
            'model': model_config['model_name'],
            'severity_color': get_severity_color(severity_label.lower().split()[0]),
            'vital_signs': {k: float(v) for k, v in data.items() if k in config.FEATURE_COLUMNS}
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html',
                         model_config=model_config,
                         severity_thresholds=config.SEVERITY_THRESHOLDS)


@app.route('/dashboard')
def dashboard():
    """EDA Dashboard with all visualizations"""
    import os
    from pathlib import Path
    
    # Get all available plots
    eda_plots = []
    model_plots = []
    
    # EDA visualizations
    if config.EDA_DIR.exists():
        for plot_file in config.EDA_DIR.glob('*.png'):
            eda_plots.append({
                'name': plot_file.stem.replace('_', ' ').title(),
                'path': f'/static/outputs/eda/{plot_file.name}'
            })
    
    # Model visualizations
    if config.PLOTS_DIR.exists():
        for plot_file in config.PLOTS_DIR.glob('*.png'):
            model_plots.append({
                'name': plot_file.stem.replace('_', ' ').title(),
                'path': f'/static/outputs/plots/{plot_file.name}'
            })
    
    # Get model comparison data if available
    comparison_data = None
    comparison_file = config.OUTPUTS_DIR / 'model_comparison.csv'
    if comparison_file.exists():
        import pandas as pd
        comparison_df = pd.read_csv(comparison_file)
        comparison_data = comparison_df.to_html(classes='table table-striped table-hover', index=False)
    
    return render_template('dashboard.html',
                         eda_plots=eda_plots,
                         model_plots=model_plots,
                         comparison_data=comparison_data,
                         model_config=model_config)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded_model is not None,
        'model_name': model_config['model_name'] if model_config else None
    })


if __name__ == '__main__':
    # Load model on startup
    load_best_model()
    
    # Run app
    print("\n" + "="*80)
    print(" "*25 + "FLASK APP STARTING")
    print("="*80)
    print(f"\n🚀 Server running at: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"📊 Model: {model_config['model_name']}")
    print(f"✨ Features: Real-time anomaly detection with explainability")
    print("\n" + "="*80 + "\n")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
