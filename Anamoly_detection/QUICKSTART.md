# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (2 minutes)

```bash
cd C:\Users\Gowthami\OneDrive\Desktop\AnomalyDetection
pip install -r requirements.txt
```

### Step 2: Train Models (5-10 minutes)

```bash
python train_all_models.py --no-tuning
```

This will:
- ✅ Preprocess data
- ✅ Generate EDA visualizations
- ✅ Train all 3 models
- ✅ Select best model

### Step 3: Run Flask App

```bash
python app.py
```

Open browser to: **http://localhost:5000**

---

## 📊 Full Training with Hyperparameter Tuning (30-60 minutes)

For best results with Optuna optimization:

```bash
python train_all_models.py
```

This includes:
- GridSearchCV for Isolation Forest (contamination tuning)
- Optuna for Autoencoder (50 trials)
- Optuna for LSTM Autoencoder (sequence_length tuning, 50 trials)

---

## 🧪 Test Individual Components

### Test Data Preprocessing
```bash
python data_preprocessing.py
```

### Generate EDA Only
```bash
python exploratory_analysis.py
```

### Test Flask API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"heart_rate\": 75, \"blood_oxygen\": 98, \"blood_pressure_systolic\": 120, \"blood_pressure_diastolic\": 80, \"glucose_level\": 95, \"body_temperature\": 98.6, \"respiratory_rate\": 16, \"activity_level\": 5, \"sleep_quality\": 7, \"stress_level\": 3, \"hrv_sdnn\": 50, \"steps_count\": 8000, \"calories_burned\": 2000}"
```

---

## 📁 Project Structure

```
AnomalyDetection/
├── config.py                    # Configuration
├── data_preprocessing.py        # Data processing
├── exploratory_analysis.py      # EDA generation
├── hyperparameter_tuning.py     # Model tuning
├── model_evaluation.py          # Evaluation
├── train_all_models.py          # Main training
├── app.py                       # Flask app
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
│
├── models/
│   ├── isolation_forest_model.py
│   ├── autoencoder_model.py
│   └── lstm_autoencoder_model.py
│
└── templates/
    ├── index.html
    └── about.html
```

---

## ✨ Key Features

- 🔍 **3 ML Models**: Isolation Forest, Autoencoder, LSTM Autoencoder
- 🎯 **Explainable AI**: SHAP values and feature contributions
- 📊 **Severity Levels**: Normal, Mild, Moderate, Severe
- 🔄 **Adaptive Thresholds**: Model drift detection
- 🎨 **Beautiful UI**: Bootstrap 5 with animations

---

## 🆘 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### PyTorch Installation Issues
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Port Already in Use
Edit `config.py` and change `FLASK_PORT = 5001`

---

## 📚 Next Steps

1. ✅ Train models
2. ✅ Explore EDA visualizations in `outputs/eda/`
3. ✅ Check model comparison in `outputs/model_comparison.csv`
4. ✅ Test Flask app at http://localhost:5000
5. ✅ Read full documentation in `README.md`

---

**Need help?** Check `README.md` for detailed documentation!
