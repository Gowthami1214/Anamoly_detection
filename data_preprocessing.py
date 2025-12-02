"""
Data Preprocessing Module for Patient Vital Signs Anomaly Detection
Handles data loading, cleaning, timestamp conversion, and scaling
CRITICAL: Keeps timestamp column in dataset, removes only from training features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

import config

def load_and_preprocess_data(scaler_type='standard', test_size=0.2, save_processed=True):
    """
    Load and preprocess the IoT health monitoring dataset
    
    Args:
        scaler_type: 'standard' or 'minmax'
        test_size: Proportion of data for testing
        save_processed: Whether to save processed data and scaler
        
    Returns:
        X_train: Training features (without timestamp)
        X_test: Test features (without timestamp)
        y_train: Training labels (health_event)
        y_test: Test labels (health_event)
        train_timestamps: Timestamps for training data
        test_timestamps: Timestamps for test data
        scaler: Fitted scaler object
        df_processed: Full processed dataframe with timestamp
    """
    
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    # Load dataset
    print(f"\n[1/7] Loading dataset from: {config.DATASET_PATH}")
    df = pd.read_csv(config.DATASET_PATH)
    print(f"   ✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert timestamp to datetime
    print("\n[2/7] Converting timestamp to datetime...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   ✓ Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check for missing values
    print("\n[3/7] Checking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() == 0:
        print("   ✓ No missing values found")
    else:
        print(f"   ⚠ Missing values found:\n{missing_counts[missing_counts > 0]}")
        print("   → Filling missing values with median...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Separate features and labels
    print("\n[4/7] Separating features and labels...")
    
    # Keep timestamp in dataframe, but exclude from features
    feature_cols = [col for col in df.columns if col not in config.EXCLUDE_COLUMNS]
    
    # Extract features (without timestamp, device_id, patient_id, health_event)
    X = df[feature_cols].copy()
    
    # Extract labels (health_event for evaluation only)
    y = df['health_event'].copy() if 'health_event' in df.columns else None
    
    # Keep timestamps separately
    timestamps = df['timestamp'].copy()
    
    print(f"   ✓ Features: {len(feature_cols)} columns")
    print(f"   ✓ Feature columns: {feature_cols}")
    
    # Train-test split (stratified if labels available)
    print(f"\n[5/7] Splitting data (test_size={test_size})...")
    if y is not None:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(len(X)), 
            test_size=test_size, 
            random_state=config.RANDOM_STATE,
            stratify=y
        )
    else:
        X_train, X_test, train_idx, test_idx = train_test_split(
            X, np.arange(len(X)),
            test_size=test_size,
            random_state=config.RANDOM_STATE
        )
        y_train, y_test = None, None
    
    # Get corresponding timestamps
    train_timestamps = timestamps.iloc[train_idx].reset_index(drop=True)
    test_timestamps = timestamps.iloc[test_idx].reset_index(drop=True)
    
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # Scale features
    print(f"\n[6/7] Scaling features using {scaler_type.upper()} scaler...")
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"   ✓ Features scaled successfully")
    
    # Save processed data and scaler
    if save_processed:
        print(f"\n[7/7] Saving processed data and scaler...")
        
        # Save scaler
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"   ✓ Scaler saved to: {config.SCALER_PATH}")
        
        # Save processed data
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_timestamps': train_timestamps,
            'test_timestamps': test_timestamps,
            'feature_columns': feature_cols
        }
        joblib.dump(processed_data, config.PROCESSED_DATA_PATH)
        print(f"   ✓ Processed data saved to: {config.PROCESSED_DATA_PATH}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Training samples: {len(X_train_scaled)}")
    print(f"  • Test samples: {len(X_test_scaled)}")
    print(f"  • Features: {len(feature_cols)}")
    if y is not None:
        print(f"  • Anomaly distribution (health_event):")
        print(f"    - Training: {y_train.value_counts().to_dict()}")
        print(f"    - Test: {y_test.value_counts().to_dict()}")
    print("=" * 60 + "\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, train_timestamps, test_timestamps, scaler


def get_feature_names():
    """Return list of feature column names"""
    return [col for col in config.FEATURE_COLUMNS if col not in config.EXCLUDE_COLUMNS]


def get_scaler(scaler_path=None):
    """
    Load saved scaler for Flask app
    
    Args:
        scaler_path: Path to saved scaler (default: config.SCALER_PATH)
        
    Returns:
        Fitted scaler object
    """
    if scaler_path is None:
        scaler_path = config.SCALER_PATH
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run preprocessing first.")
    
    return joblib.load(scaler_path)


def load_processed_data():
    """
    Load previously saved processed data
    
    Returns:
        Dictionary containing X_train, X_test, y_train, y_test, timestamps, feature_columns
    """
    if not config.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {config.PROCESSED_DATA_PATH}. Please run preprocessing first.")
    
    return joblib.dump(config.PROCESSED_DATA_PATH)


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test, train_ts, test_ts, scaler = load_and_preprocess_data(
        scaler_type=config.SCALER_TYPE,
        test_size=config.TEST_SIZE
    )
    
    print("\n✅ Preprocessing test successful!")
    print(f"\nFeature names: {get_feature_names()}")
