# Unsupervised Detection of Anomalies in Patient Vital Signs using Machine Learning

A complete end-to-end machine learning system for detecting anomalies in patient vital signs using unsupervised learning techniques. The system provides real-time analysis with explainable AI features and a user-friendly web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This system implements three advanced anomaly detection models:
- **Isolation Forest** with SHAP explainability
- **Autoencoder** with per-feature reconstruction error analysis
- **LSTM Autoencoder** with time-step error heatmaps

### Key Features

✨ **Explainable AI**: SHAP values and feature contribution analysis  
📊 **Severity Categorization**: Normal, Mild, Moderate, and Severe levels  
🔄 **Adaptive Thresholds**: Automatic recalibration for model drift  
⚡ **Real-time Detection**: Instant analysis via Flask web interface  
🎨 **Beautiful UI**: Modern Bootstrap 5 interface with animations  
🔧 **Hyperparameter Tuning**: Optuna and GridSearchCV optimization  

## 📁 Project Structure

```
AnomalyDetection/
├── config.py                          # Centralized configuration
├── data_preprocessing.py              # Data loading and preprocessing
├── exploratory_analysis.py            # EDA generation
├── hyperparameter_tuning.py           # Model tuning with Optuna
├── model_evaluation.py                # Evaluation and comparison
├── train_all_models.py                # Main training pipeline
├── app.py                             # Flask web application
├── requirements.txt                   # Python dependencies
│
├── models/
│   ├── isolation_forest_model.py      # Isolation Forest with SHAP
│   ├── autoencoder_model.py           # Fully connected Autoencoder
│   └── lstm_autoencoder_model.py      # LSTM Autoencoder
│
├── templates/
│   ├── index.html                     # Main prediction interface
│   └── about.html                     # About page
│
├── data/                              # Processed data (generated)
├── models/saved/                      # Trained models (generated)
└── outputs/                           # Visualizations and reports (generated)
    ├── eda/                           # EDA visualizations
    └── plots/                         # Model visualizations
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**
```bash
cd C:\Users\Gowthami\OneDrive\Desktop\AnomalyDetection
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The system uses the **IoT Health Monitoring Dataset** (`iot_health_monitoring_dataset.csv`) with 5,094 samples and 13 vital sign features:

- Heart Rate (bpm)
- Blood Oxygen (SpO2 %)
- Blood Pressure (Systolic & Diastolic)
- Glucose Level (mg/dL)
- Body Temperature (°F)
- Respiratory Rate (breaths/min)
- Activity Level, Sleep Quality, Stress Level (0-10 scale)
- HRV SDNN (heart rate variability)
- Steps Count, Calories Burned

## 🎓 Training the Models

### Quick Start (Default Parameters)

```bash
python train_all_models.py --no-tuning
```

This will:
1. ✅ Preprocess data with timestamp preservation
2. ✅ Generate comprehensive EDA visualizations
3. ✅ Train all three models with default parameters
4. ✅ Evaluate and compare models
5. ✅ Select and save the best model

**Estimated time**: ~5-10 minutes

### Full Training with Hyperparameter Tuning

```bash
python train_all_models.py
```

This includes:
- GridSearchCV for Isolation Forest (contamination tuning)
- Optuna optimization for Autoencoder (50 trials)
- Optuna optimization for LSTM Autoencoder (sequence_length tuning, 50 trials)

**Estimated time**: ~30-60 minutes depending on hardware

### Training Options

```bash
python train_all_models.py --no-tuning    # Skip hyperparameter tuning
python train_all_models.py --no-eda       # Skip EDA generation
```

## 🌐 Running the Flask Web Application

After training, start the Flask server:

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

### API Endpoints

#### `POST /predict`
Predict anomaly for patient vital signs

**Request (JSON):**
```json
{
  "heart_rate": 75.0,
  "blood_oxygen": 98.0,
  "blood_pressure_systolic": 120.0,
  "blood_pressure_diastolic": 80.0,
  "glucose_level": 95.0,
  "body_temperature": 98.6,
  "respiratory_rate": 16.0,
  "activity_level": 5.0,
  "sleep_quality": 7.0,
  "stress_level": 3.0,
  "hrv_sdnn": 50.0,
  "steps_count": 8000.0,
  "calories_burned": 2000.0
}
```

**Response:**
```json
{
  "success": true,
  "anomaly_score": 0.2345,
  "is_anomaly": 0,
  "severity": "normal",
  "severity_color": "#28a745",
  "top_contributing_features": [
    {
      "feature": "Heart Rate",
      "contribution": 0.15,
      "value": 75.0
    }
  ],
  "explanation": "All vital signs are within normal ranges. No immediate concerns detected.",
  "timestamp": "2025-12-02T10:50:00",
  "model_name": "Isolation Forest"
}
```

#### `GET /health`
Health check endpoint

#### `GET /about`
Information about the system

## 📈 Model Comparison

The system automatically evaluates all three models and selects the best based on:
- **F1-Score** (if ground truth available)
- **ROC-AUC** (if ground truth available)
- **Silhouette Score** (clustering quality)

Results are saved to `outputs/model_comparison.csv` and visualized in `outputs/plots/model_comparison.png`.

## 🎨 Severity Categorization

| Severity | Score Range | Color | Description |
|----------|------------|-------|-------------|
| **Normal** | 0.0 - 0.5 | 🟢 Green | All vital signs within expected ranges |
| **Mild** | 0.5 - 0.7 | 🟡 Yellow | Minor deviations, monitor patient |
| **Moderate** | 0.7 - 0.9 | 🟠 Orange | Significant anomalies, medical attention recommended |
| **Severe** | 0.9 - 1.0 | 🔴 Red | Critical anomalies, immediate intervention may be required |

## 🔍 Explainability Features

### Isolation Forest
- **SHAP TreeExplainer**: Provides feature importance for each prediction
- Shows which vital signs contributed most to the anomaly score

### Autoencoder
- **Per-Feature Reconstruction Error**: Identifies which features were hardest to reconstruct
- Highlights specific vital signs that deviate from learned patterns

### LSTM Autoencoder
- **Time-Step Error Heatmaps**: Shows anomalies across temporal sequences
- Identifies when in the sequence anomalies occurred

## 🔄 Adaptive Threshold Recalibration

The system includes a `update_threshold()` function to handle model drift:

```python
from model_evaluation import update_threshold

new_threshold, drift_detected = update_threshold(model, new_data, percentile=95)
```

This automatically detects if the data distribution has shifted and recalibrates thresholds accordingly.

## 📊 Generated Outputs

After training, the following outputs are generated:

### EDA Visualizations (`outputs/eda/`)
- `feature_distributions.png` - Distribution plots for all vital signs
- `correlation_heatmap.png` - Feature correlation analysis
- `timeseries_vitals.png` - Time-series plots for key vitals
- `outlier_boxplots.png` - Box plots for outlier detection
- `pairwise_scatter.png` - Scatter plots for correlated features
- `statistical_summary.txt` - Comprehensive statistical report

### Model Visualizations (`outputs/plots/`)
- `isolation_forest_anomalies.png` - Anomaly visualization
- `autoencoder_anomalies.png` - Reconstruction error plots
- `lstm_autoencoder_anomalies.png` - Sequence anomaly plots
- `model_comparison.png` - Side-by-side model comparison

### Evaluation Results (`outputs/`)
- `model_comparison.csv` - Detailed metrics comparison
- `training_summary.txt` - Complete training report

## 🛠️ Configuration

All hyperparameters and settings can be modified in `config.py`:

```python
# Severity thresholds
SEVERITY_THRESHOLDS = {
    'normal': (0.0, 0.5),
    'mild': (0.5, 0.7),
    'moderate': (0.7, 0.9),
    'severe': (0.9, 1.0)
}

# Isolation Forest tuning
ISOLATION_FOREST_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'contamination': [0.01, 0.03, 0.05, 0.1]
}

# LSTM sequence length tuning
LSTM_OPTUNA_PARAMS = {
    'sequence_length': [5, 10, 20, 30]
}
```

## 🧪 Testing

Test the preprocessing pipeline:
```bash
python data_preprocessing.py
```

Test EDA generation:
```bash
python exploratory_analysis.py
```

Test individual models:
```bash
python -c "from models.isolation_forest_model import IsolationForestDetector; print('✓ Isolation Forest loaded')"
```

## 📝 Example Usage

### Python API

```python
import pandas as pd
from models.isolation_forest_model import IsolationForestDetector
from data_preprocessing import get_scaler

# Load model and scaler
model = IsolationForestDetector.load_model('models/saved/isolation_forest.pkl')
scaler = get_scaler()

# Prepare input
vital_signs = pd.DataFrame([{
    'heart_rate': 180,  # Abnormally high
    'blood_oxygen': 85,  # Low
    'blood_pressure_systolic': 160,
    # ... other features
}])

# Scale and predict
vital_signs_scaled = scaler.transform(vital_signs)
scores, labels, severity = model.predict(vital_signs_scaled)

# Get explanations
contributions = model.get_feature_contributions(vital_signs_scaled, top_n=5)

print(f"Anomaly Score: {scores[0]:.4f}")
print(f"Severity: {severity[0]}")
print(f"Top Contributors: {[c['feature'] for c in contributions]}")
```

## 🤝 Contributing

This project was created as a complete end-to-end ML system. Feel free to:
- Add new anomaly detection models
- Improve the UI/UX
- Enhance explainability features
- Add more evaluation metrics

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: IoT Health Monitoring Dataset
- SHAP library for explainable AI
- Optuna for hyperparameter optimization
- Bootstrap 5 for beautiful UI

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for healthcare AI applications**
