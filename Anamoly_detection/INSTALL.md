# Installation Guide for Windows

## ✅ Virtual Environment Created!

The virtual environment `venv` has been created in your project directory.

---

## 🚀 Quick Installation (Recommended)

### Option 1: Use the Installation Script

Simply double-click `install.bat` or run:

```bash
install.bat
```

This will automatically:
1. Activate the virtual environment
2. Upgrade pip
3. Install all dependencies in the correct order

---

## 📝 Manual Installation (Alternative)

If you prefer to install manually:

### Step 1: Activate Virtual Environment

```bash
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 3: Install Core Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask joblib scipy
```

### Step 4: Install PyTorch (CPU version)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install Remaining Packages

```bash
pip install optuna shap plotly kaleido
```

---

## ✅ Verify Installation

After installation, verify everything is working:

```bash
python -c "import pandas, numpy, sklearn, torch, flask, optuna, shap; print('✓ All packages installed successfully!')"
```

---

## 🎯 Next Steps

### 1. Train the Models (5-10 minutes)

```bash
python train_all_models.py --no-tuning
```

### 2. Run the Flask App

```bash
python app.py
```

### 3. Open Your Browser

Navigate to: **http://localhost:5000**

---

## 🔧 Troubleshooting

### If you get "venv\Scripts\activate is not recognized"

Make sure you're in the project directory:
```bash
cd C:\Users\Gowthami\OneDrive\Desktop\AnomalyDetection
```

### If PyTorch installation fails

Try installing the CPU-only version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### If SHAP installation fails

SHAP might need Microsoft C++ Build Tools. You can:
1. Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Or skip SHAP for now (Isolation Forest will still work with basic feature importance)

---

## 💡 Tips

- Always activate the virtual environment before running any Python commands
- To deactivate the virtual environment, simply type: `deactivate`
- The virtual environment is isolated - packages installed here won't affect your system Python

---

## 📚 Quick Reference

```bash
# Activate venv
venv\Scripts\activate

# Train models (quick)
python train_all_models.py --no-tuning

# Train with hyperparameter tuning (30-60 min)
python train_all_models.py

# Run Flask app
python app.py

# Deactivate venv
deactivate
```

---

**Need help?** Check `README.md` for full documentation!
