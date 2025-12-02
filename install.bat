@echo off
echo ========================================
echo Virtual Environment Setup
echo ========================================
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 3: Installing dependencies...
pip install pandas numpy scikit-learn matplotlib seaborn flask joblib scipy

echo.
echo Step 4: Installing PyTorch (CPU version)...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 5: Installing remaining packages...
pip install optuna shap plotly kaleido

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To run the training pipeline:
echo   python train_all_models.py --no-tuning
echo.
echo To start the Flask app:
echo   python app.py
echo.
pause
