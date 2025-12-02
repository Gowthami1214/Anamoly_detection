"""
Hyperparameter Tuning Module
Implements GridSearchCV for Isolation Forest and Optuna for neural networks
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.append(str(Path(__file__).parent))
import config
from models.autoencoder_model import AutoencoderDetector
from models.lstm_autoencoder_model import LSTMAutoencoderDetector

def tune_isolation_forest(X_train, y_train=None):
    """
    Tune Isolation Forest hyperparameters using GridSearchCV
    
    Args:
        X_train: Training data
        y_train: Labels (optional, for scoring)
        
    Returns:
        best_params: Dictionary of best parameters
        best_model: Best trained model
        results: GridSearch results
    """
    print("\n" + "=" * 60)
    print("TUNING ISOLATION FOREST HYPERPARAMETERS")
    print("=" * 60)
    
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    
    print(f"\n[1/3] Setting up GridSearchCV...")
    print(f"   • Parameter grid:")
    for param, values in config.ISOLATION_FOREST_PARAM_GRID.items():
        print(f"     - {param}: {values}")
    
    # Create base model
    base_model = IsolationForest(random_state=config.RANDOM_STATE, n_jobs=-1)
    
    # Define scoring function
    def anomaly_score(estimator, X):
        """Custom scoring: lower is better (more negative scores for anomalies)"""
        scores = estimator.decision_function(X)
        return -np.mean(scores)  # Negative because GridSearch maximizes
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=config.ISOLATION_FOREST_PARAM_GRID,
        scoring=anomaly_score,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n[2/3] Running GridSearchCV...")
    print(f"   • Total combinations: {np.prod([len(v) for v in config.ISOLATION_FOREST_PARAM_GRID.values()])}")
    print(f"   • Cross-validation folds: 3")
    
    grid_search.fit(X_train)
    
    print(f"\n[3/3] GridSearchCV complete!")
    print(f"\n   ✓ Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"     - {param}: {value}")
    print(f"\n   ✓ Best score: {grid_search.best_score_:.6f}")
    
    # Save results
    results_path = config.OUTPUTS_DIR / "isolation_forest_tuning_results.pkl"
    joblib.dump({
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }, results_path)
    print(f"\n   ✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("ISOLATION FOREST TUNING COMPLETE")
    print("=" * 60 + "\n")
    
    return grid_search.best_params_, grid_search.best_estimator_, grid_search.cv_results_


def tune_autoencoder(X_train, X_val=None, n_trials=50):
    """
    Tune Autoencoder hyperparameters using Optuna
    
    Args:
        X_train: Training data
        X_val: Validation data (optional)
        n_trials: Number of Optuna trials
        
    Returns:
        best_params: Dictionary of best parameters
        study: Optuna study object
    """
    print("\n" + "=" * 60)
    print("TUNING AUTOENCODER HYPERPARAMETERS")
    print("=" * 60)
    
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train = X_train.values
    else:
        feature_names = None
    
    # Split validation if not provided
    if X_val is None:
        val_size = int(len(X_train) * config.VALIDATION_SPLIT)
        X_val = X_train[-val_size:]
        X_train = X_train[:-val_size]
    elif isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    
    print(f"\n[1/3] Setting up Optuna study...")
    print(f"   • Number of trials: {n_trials}")
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Validation samples: {X_val.shape[0]}")
    
    def objective(trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        hidden_dim_1 = trial.suggest_int('hidden_dim_1', *config.AUTOENCODER_OPTUNA_PARAMS['hidden_dim_1'])
        hidden_dim_2 = trial.suggest_int('hidden_dim_2', *config.AUTOENCODER_OPTUNA_PARAMS['hidden_dim_2'])
        hidden_dim_3 = trial.suggest_int('hidden_dim_3', *config.AUTOENCODER_OPTUNA_PARAMS['hidden_dim_3'])
        learning_rate = trial.suggest_float('learning_rate', *config.AUTOENCODER_OPTUNA_PARAMS['learning_rate'], log=True)
        batch_size = trial.suggest_categorical('batch_size', config.AUTOENCODER_OPTUNA_PARAMS['batch_size'])
        dropout_rate = trial.suggest_float('dropout_rate', *config.AUTOENCODER_OPTUNA_PARAMS['dropout_rate'])
        
        # Create model with suggested parameters
        params = {
            'encoding_dims': [hidden_dim_1, hidden_dim_2, hidden_dim_3],
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'epochs': 50,  # Reduced for tuning
            'patience': 5
        }
        
        detector = AutoencoderDetector(**params)
        
        try:
            # Train model
            detector.train(X_train, feature_names=feature_names, X_val=X_val)
            
            # Return validation loss
            return min(detector.val_losses)
        except Exception as e:
            print(f"   ⚠ Trial failed: {e}")
            return float('inf')
    
    # Create and run study
    print(f"\n[2/3] Running Optuna optimization...")
    study = optuna.create_study(direction='minimize', study_name='autoencoder_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[3/3] Optimization complete!")
    print(f"\n   ✓ Best parameters:")
    for param, value in study.best_params.items():
        print(f"     - {param}: {value}")
    print(f"\n   ✓ Best validation loss: {study.best_value:.6f}")
    
    # Save results
    results_path = config.OUTPUTS_DIR / "autoencoder_tuning_results.pkl"
    joblib.dump({
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }, results_path)
    print(f"\n   ✓ Results saved to: {results_path}")
    
    # Visualize optimization history
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(config.PLOTS_DIR / "autoencoder_optimization_history.png"))
        print(f"   ✓ Optimization history saved")
        
        fig = plot_param_importances(study)
        fig.write_image(str(config.PLOTS_DIR / "autoencoder_param_importance.png"))
        print(f"   ✓ Parameter importance saved")
    except:
        print(f"   ⚠ Could not save Optuna visualizations (install kaleido: pip install kaleido)")
    
    print("\n" + "=" * 60)
    print("AUTOENCODER TUNING COMPLETE")
    print("=" * 60 + "\n")
    
    # Convert best params to model format
    best_params = {
        'encoding_dims': [
            study.best_params['hidden_dim_1'],
            study.best_params['hidden_dim_2'],
            study.best_params['hidden_dim_3']
        ],
        'learning_rate': study.best_params['learning_rate'],
        'batch_size': study.best_params['batch_size'],
        'dropout_rate': study.best_params['dropout_rate'],
        'epochs': config.AUTOENCODER_DEFAULT['epochs'],
        'patience': config.AUTOENCODER_DEFAULT['patience']
    }
    
    return best_params, study


def tune_lstm_autoencoder(X_train, X_val=None, n_trials=50):
    """
    Tune LSTM Autoencoder hyperparameters using Optuna
    Includes sequence_length tuning as requested
    
    Args:
        X_train: Training data
        X_val: Validation data (optional)
        n_trials: Number of Optuna trials
        
    Returns:
        best_params: Dictionary of best parameters
        study: Optuna study object
    """
    print("\n" + "=" * 60)
    print("TUNING LSTM AUTOENCODER HYPERPARAMETERS")
    print("=" * 60)
    
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train = X_train.values
    else:
        feature_names = None
    
    # Split validation if not provided
    if X_val is None:
        val_size = int(len(X_train) * config.VALIDATION_SPLIT)
        X_val = X_train[-val_size:]
        X_train = X_train[:-val_size]
    elif isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    
    print(f"\n[1/3] Setting up Optuna study...")
    print(f"   • Number of trials: {n_trials}")
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Validation samples: {X_val.shape[0]}")
    print(f"   • Tuning sequence_length: {config.LSTM_OPTUNA_PARAMS['sequence_length']}")
    
    def objective(trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        lstm_units = trial.suggest_int('lstm_units', *config.LSTM_OPTUNA_PARAMS['lstm_units'])
        num_layers = trial.suggest_categorical('num_layers', config.LSTM_OPTUNA_PARAMS['num_layers'])
        sequence_length = trial.suggest_categorical('sequence_length', config.LSTM_OPTUNA_PARAMS['sequence_length'])
        learning_rate = trial.suggest_float('learning_rate', *config.LSTM_OPTUNA_PARAMS['learning_rate'], log=True)
        batch_size = trial.suggest_categorical('batch_size', config.LSTM_OPTUNA_PARAMS['batch_size'])
        dropout_rate = trial.suggest_float('dropout_rate', *config.LSTM_OPTUNA_PARAMS['dropout_rate'])
        
        # Create model with suggested parameters
        params = {
            'lstm_units': lstm_units,
            'num_layers': num_layers,
            'sequence_length': sequence_length,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'epochs': 50,  # Reduced for tuning
            'patience': 5
        }
        
        detector = LSTMAutoencoderDetector(**params)
        
        try:
            # Train model
            detector.train(X_train, feature_names=feature_names, X_val=X_val)
            
            # Return validation loss
            return min(detector.val_losses)
        except Exception as e:
            print(f"   ⚠ Trial failed: {e}")
            return float('inf')
    
    # Create and run study
    print(f"\n[2/3] Running Optuna optimization...")
    study = optuna.create_study(direction='minimize', study_name='lstm_autoencoder_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[3/3] Optimization complete!")
    print(f"\n   ✓ Best parameters:")
    for param, value in study.best_params.items():
        print(f"     - {param}: {value}")
    print(f"\n   ✓ Best validation loss: {study.best_value:.6f}")
    
    # Save results
    results_path = config.OUTPUTS_DIR / "lstm_autoencoder_tuning_results.pkl"
    joblib.dump({
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }, results_path)
    print(f"\n   ✓ Results saved to: {results_path}")
    
    # Visualize optimization history
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(config.PLOTS_DIR / "lstm_optimization_history.png"))
        print(f"   ✓ Optimization history saved")
        
        fig = plot_param_importances(study)
        fig.write_image(str(config.PLOTS_DIR / "lstm_param_importance.png"))
        print(f"   ✓ Parameter importance saved")
    except:
        print(f"   ⚠ Could not save Optuna visualizations (install kaleido: pip install kaleido)")
    
    print("\n" + "=" * 60)
    print("LSTM AUTOENCODER TUNING COMPLETE")
    print("=" * 60 + "\n")
    
    # Return best params in model format
    best_params = {
        'lstm_units': study.best_params['lstm_units'],
        'num_layers': study.best_params['num_layers'],
        'sequence_length': study.best_params['sequence_length'],
        'learning_rate': study.best_params['learning_rate'],
        'batch_size': study.best_params['batch_size'],
        'dropout_rate': study.best_params['dropout_rate'],
        'epochs': config.LSTM_AUTOENCODER_DEFAULT['epochs'],
        'patience': config.LSTM_AUTOENCODER_DEFAULT['patience']
    }
    
    return best_params, study


if __name__ == "__main__":
    print("Hyperparameter Tuning module loaded successfully!")
    print("\nAvailable functions:")
    print("  • tune_isolation_forest(X_train, y_train)")
    print("  • tune_autoencoder(X_train, X_val, n_trials)")
    print("  • tune_lstm_autoencoder(X_train, X_val, n_trials)")
