"""
Main Training Script
Orchestrates the entire anomaly detection pipeline:
1. Data preprocessing
2. EDA generation
3. Hyperparameter tuning
4. Model training
5. Evaluation and comparison
6. Save best model for Flask app
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import config
from data_preprocessing import load_and_preprocess_data
from exploratory_analysis import generate_eda_report
from hyperparameter_tuning import tune_isolation_forest, tune_autoencoder, tune_lstm_autoencoder
from models.isolation_forest_model import IsolationForestDetector
from models.autoencoder_model import AutoencoderDetector
from models.lstm_autoencoder_model import LSTMAutoencoderDetector
from model_evaluation import evaluate_model, compare_models, select_best_model

def main(run_tuning=True, run_eda=True):
    """
    Main training pipeline
    
    Args:
        run_tuning: Whether to run hyperparameter tuning (takes time)
        run_eda: Whether to generate EDA visualizations
    """
    print("\n" + "="*80)
    print(" "*20 + "ANOMALY DETECTION TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # ========== STEP 1: DATA PREPROCESSING ==========
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 80)
    
    X_train, X_test, y_train, y_test, train_timestamps, test_timestamps, scaler = load_and_preprocess_data(
        scaler_type=config.SCALER_TYPE,
        test_size=config.TEST_SIZE
    )
    
    feature_names = X_train.columns.tolist()
    
    # ========== STEP 2: EXPLORATORY DATA ANALYSIS ==========
    if run_eda:
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("-" * 80)
        
        # Load original data for EDA
        df = pd.read_csv(config.DATASET_PATH)
        generate_eda_report(df)
    
    # ========== STEP 3: HYPERPARAMETER TUNING ==========
    if run_tuning:
        print("\n" + "="*80)
        print("STEP 3: HYPERPARAMETER TUNING")
        print("-" * 80)
        
        print("\n[3.1] Tuning Isolation Forest...")
        if_best_params, _, _ = tune_isolation_forest(X_train, y_train)
        
        print("\n[3.2] Tuning Autoencoder...")
        ae_best_params, _ = tune_autoencoder(X_train, n_trials=30)  # Reduced trials for demo
        
        print("\n[3.3] Tuning LSTM Autoencoder...")
        lstm_best_params, _ = tune_lstm_autoencoder(X_train, n_trials=30)  # Reduced trials for demo
        
    else:
        print("\n" + "="*80)
        print("STEP 3: USING DEFAULT HYPERPARAMETERS (Tuning Skipped)")
        print("-" * 80)
        
        if_best_params = config.ISOLATION_FOREST_DEFAULT
        ae_best_params = config.AUTOENCODER_DEFAULT
        lstm_best_params = config.LSTM_AUTOENCODER_DEFAULT
    
    # ========== STEP 4: MODEL TRAINING ==========
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING")
    print("-" * 80)
    
    # 4.1 Train Isolation Forest
    print("\n[4.1] Training Isolation Forest with SHAP...")
    if_detector = IsolationForestDetector(**if_best_params)
    if_detector.train(X_train, feature_names=feature_names)
    if_detector.save_model(config.MODELS_DIR / "isolation_forest.pkl")
    
    # 4.2 Train Autoencoder
    print("\n[4.2] Training Autoencoder with Per-Feature Explainability...")
    ae_detector = AutoencoderDetector(**ae_best_params)
    ae_detector.train(X_train, feature_names=feature_names)
    ae_detector.save_model(config.MODELS_DIR / "autoencoder.pkl")
    
    # 4.3 Train LSTM Autoencoder
    print("\n[4.3] Training LSTM Autoencoder with Time-Step Heatmaps...")
    lstm_detector = LSTMAutoencoderDetector(**lstm_best_params)
    lstm_detector.train(X_train, feature_names=feature_names, timestamps=train_timestamps)
    lstm_detector.save_model(config.MODELS_DIR / "lstm_autoencoder.pkl")
    
    # ========== STEP 5: MODEL EVALUATION ==========
    print("\n" + "="*80)
    print("STEP 5: MODEL EVALUATION")
    print("-" * 80)
    
    model_results = {}
    
    # 5.1 Evaluate Isolation Forest
    print("\n[5.1] Evaluating Isolation Forest...")
    if_scores, if_labels, if_severity = if_detector.predict(X_test)
    if_metrics = evaluate_model(
        model_name="Isolation Forest",
        scores=if_scores,
        labels=if_labels,
        severity=if_severity,
        y_true=y_test
    )
    model_results["Isolation Forest"] = if_metrics
    
    # Visualize
    if_detector.visualize_anomalies(
        X_test, test_timestamps, if_scores, if_labels,
        save_path=config.PLOTS_DIR / "isolation_forest_anomalies.png"
    )
    
    # 5.2 Evaluate Autoencoder
    print("\n[5.2] Evaluating Autoencoder...")
    ae_scores, ae_labels, ae_severity = ae_detector.predict(X_test)
    ae_metrics = evaluate_model(
        model_name="Autoencoder",
        scores=ae_scores,
        labels=ae_labels,
        severity=ae_severity,
        y_true=y_test
    )
    model_results["Autoencoder"] = ae_metrics
    
    # Visualize
    ae_detector.visualize_anomalies(
        X_test, test_timestamps, ae_scores, ae_labels,
        save_path=config.PLOTS_DIR / "autoencoder_anomalies.png"
    )
    
    # 5.3 Evaluate LSTM Autoencoder
    print("\n[5.3] Evaluating LSTM Autoencoder...")
    lstm_scores, lstm_labels, lstm_severity, lstm_timestamps = lstm_detector.predict(X_test, test_timestamps)
    
    # Align y_test with LSTM sequences
    if y_test is not None:
        seq_len = lstm_detector.params['sequence_length']
        y_test_lstm = y_test.iloc[seq_len-1:].reset_index(drop=True) if isinstance(y_test, pd.Series) else y_test[seq_len-1:]
    else:
        y_test_lstm = None
    
    lstm_metrics = evaluate_model(
        model_name="LSTM Autoencoder",
        scores=lstm_scores,
        labels=lstm_labels,
        severity=lstm_severity,
        y_true=y_test_lstm
    )
    model_results["LSTM Autoencoder"] = lstm_metrics
    
    # Visualize
    lstm_detector.visualize_anomalies(
        X_test, lstm_timestamps, lstm_scores, lstm_labels,
        save_path=config.PLOTS_DIR / "lstm_autoencoder_anomalies.png"
    )
    
    # ========== STEP 6: MODEL COMPARISON ==========
    print("\n" + "="*80)
    print("STEP 6: MODEL COMPARISON")
    print("-" * 80)
    
    comparison_df = compare_models(
        model_results,
        save_path=config.PLOTS_DIR / "model_comparison.png"
    )
    
    # Save comparison table
    comparison_df.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    print(f"✓ Comparison table saved to: {config.OUTPUTS_DIR / 'model_comparison.csv'}\n")
    
    # ========== STEP 7: SELECT BEST MODEL ==========
    print("\n" + "="*80)
    print("STEP 7: SELECT BEST MODEL")
    print("-" * 80)
    
    best_model_name = select_best_model(model_results)
    
    # Save best model for Flask app
    if best_model_name == "Isolation Forest":
        best_model = if_detector
    elif best_model_name == "Autoencoder":
        best_model = ae_detector
    else:
        best_model = lstm_detector
    
    # Save best model reference
    best_model_config = {
        'model_name': best_model_name,
        'model_path': config.MODELS_DIR / f"{best_model_name.lower().replace(' ', '_')}.pkl",
        'scaler_path': config.SCALER_PATH,
        'feature_names': feature_names,
        'metrics': model_results[best_model_name]
    }
    
    joblib.dump(best_model_config, config.BEST_MODEL_PATH)
    print(f"✓ Best model configuration saved to: {config.BEST_MODEL_PATH}\n")
    
    # ========== STEP 8: GENERATE SUMMARY REPORT ==========
    print("\n" + "="*80)
    print("STEP 8: GENERATING SUMMARY REPORT")
    print("-" * 80)
    
    with open(config.OUTPUTS_DIR / "training_summary.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "ANOMALY DETECTION TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Feature names: {', '.join(feature_names)}\n\n")
        
        f.write("HYPERPARAMETERS\n")
        f.write("-"*80 + "\n")
        f.write(f"\nIsolation Forest:\n{if_best_params}\n")
        f.write(f"\nAutoencoder:\n{ae_best_params}\n")
        f.write(f"\nLSTM Autoencoder:\n{lstm_best_params}\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("BEST MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Selected Model: {best_model_name}\n")
        f.write(f"F1-Score: {model_results[best_model_name].get('f1_score', 'N/A')}\n")
        f.write(f"ROC-AUC: {model_results[best_model_name].get('roc_auc', 'N/A')}\n")
        f.write(f"Anomaly Rate: {model_results[best_model_name]['anomaly_rate']:.2f}%\n\n")
        
        f.write("SEVERITY DISTRIBUTION (Best Model)\n")
        f.write("-"*80 + "\n")
        for severity in ['normal', 'mild', 'moderate', 'severe']:
            pct = model_results[best_model_name].get(f'severity_{severity}_pct', 0)
            f.write(f"{severity.capitalize()}: {pct:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TRAINING COMPLETE\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Training summary saved to: {config.OUTPUTS_DIR / 'training_summary.txt'}\n")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print(" "*25 + "TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print("\n📊 Generated Outputs:")
    print(f"   • EDA Visualizations: {config.EDA_DIR}")
    print(f"   • Model Visualizations: {config.PLOTS_DIR}")
    print(f"   • Trained Models: {config.MODELS_DIR}")
    print(f"   • Evaluation Results: {config.OUTPUTS_DIR}")
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   • Model Path: {best_model_config['model_path']}")
    print(f"   • F1-Score: {model_results[best_model_name].get('f1_score', 'N/A')}")
    print(f"   • ROC-AUC: {model_results[best_model_name].get('roc_auc', 'N/A')}")
    print("\n✅ Ready for Flask deployment!")
    print("   Run: python app.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--no-eda', action='store_true', help='Skip EDA generation')
    
    args = parser.parse_args()
    
    main(
        run_tuning=not args.no_tuning,
        run_eda=not args.no_eda
    )
