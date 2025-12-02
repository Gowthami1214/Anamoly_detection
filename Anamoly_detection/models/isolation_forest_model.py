"""
Isolation Forest Anomaly Detection Model with SHAP Explainability
Implements tree-based anomaly detection with feature importance analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector with SHAP explainability
    """
    
    def __init__(self, **params):
        """
        Initialize Isolation Forest detector
        
        Args:
            **params: Parameters for IsolationForest (n_estimators, contamination, etc.)
        """
        # Use default parameters if not provided
        self.params = {**config.ISOLATION_FOREST_DEFAULT, **params}
        self.model = IsolationForest(**self.params)
        self.explainer = None
        self.feature_names = None
        self.min_score = None
        self.max_score = None
        
    def train(self, X_train, feature_names=None):
        """
        Train the Isolation Forest model
        
        Args:
            X_train: Training data (numpy array or DataFrame)
            feature_names: List of feature names
        """
        print("\n" + "=" * 60)
        print("TRAINING ISOLATION FOREST MODEL")
        print("=" * 60)
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        print(f"\n[1/3] Training Isolation Forest...")
        print(f"   • Samples: {X_train.shape[0]}")
        print(f"   • Features: {X_train.shape[1]}")
        print(f"   • Parameters: {self.params}")
        
        self.model.fit(X_train)
        print(f"   ✓ Model trained successfully")
        
        # Calculate min/max scores for normalization
        # Decision function: lower is more anomalous
        raw_scores = self.model.decision_function(X_train)
        self.min_score = raw_scores.min()
        self.max_score = raw_scores.max()
        print(f"   ✓ Score range established: [{self.min_score:.4f}, {self.max_score:.4f}]")
        
        # Initialize SHAP explainer
        print(f"\n[2/3] Initializing SHAP explainer...")
        # Use a subset of training data for SHAP background
        background_size = min(config.SHAP_SAMPLES, len(X_train))
        background_indices = np.random.choice(len(X_train), background_size, replace=False)
        background_data = X_train[background_indices]
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            print(f"   ✓ SHAP explainer initialized")
        except Exception as e:
            print(f"   ⚠ Could not initialize SHAP explainer: {e}")
            self.explainer = None
        
        print("\n" + "=" * 60)
        print("ISOLATION FOREST TRAINING COMPLETE")
        print("=" * 60 + "\n")
    
    def predict(self, X):
        """
        Predict anomaly scores and labels
        
        Args:
            X: Input data
            
        Returns:
            scores: Normalized anomaly scores (0-1)
            labels: Binary labels (0=normal, 1=anomaly)
            severity_info: List of dictionaries with severity details
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get decision function scores (lower = more anomalous)
        # Range is typically [-0.5, 0.5] where negative is anomaly
        raw_scores = self.model.decision_function(X)
        
        # Normalize scores to 0-1 range (higher = more anomalous)
        # We use the min/max from training to scale
        # If raw_score is min_score (most anomalous) -> 1.0
        # If raw_score is max_score (most normal) -> 0.0
        
        if self.max_score == self.min_score:
            scores = np.zeros_like(raw_scores)
        else:
            # Formula: (max - x) / (max - min)
            scores = (self.max_score - raw_scores) / (self.max_score - self.min_score)
            
        # Clip to 0-1 range
        scores = np.clip(scores, 0.0, 1.0)
        
        # Get binary predictions from model
        # -1 is anomaly, 1 is normal
        model_preds = self.model.predict(X)
        labels = np.where(model_preds == -1, 1, 0)
        
        # Categorize severity
        severity_info = self.categorize_severity(scores)
        
        return scores, labels, severity_info
    
    def categorize_severity(self, scores):
        """
        Categorize anomaly scores into severity levels
        
        Args:
            scores (array-like): Normalized anomaly scores (0-1)
            
        Returns:
            list: List of dictionaries with 'label' and 'class'
        """
        severity_results = []
        
        for score in scores:
            # Default
            label = 'Normal'
            class_id = 0
            
            # Check thresholds (Strict adherence to user requirements)
            if 0.00 <= score <= 0.30:
                label = 'Normal'
                class_id = 0
            elif 0.30 < score <= 0.60:
                label = 'Borderline Anomaly'
                class_id = 1
            elif 0.60 < score <= 0.85:
                label = 'Highly Abnormal Anomaly'
                class_id = 2
            elif 0.85 < score <= 1.00:
                label = 'Extreme Abnormal Anomaly'
                class_id = 3
            
            severity_results.append({
                'severity_label': label,
                'severity_class': class_id
            })
            
        return severity_results
    
    def get_feature_contributions(self, X, top_n=5):
        """
        Get top contributing features using SHAP values
        
        Args:
            X: Input data (single sample or batch)
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with feature contributions
        """
        if self.explainer is None:
            return []
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Get SHAP values
        try:
            shap_values = self.explainer.shap_values(X)
            # Handle case where shap_values might be a list (though TreeExplainer usually returns array for IF)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return []
            
        contributions = []
        for i in range(len(X)):
            # Get absolute SHAP values for this sample
            # We take absolute values to show magnitude of impact
            sample_shap = np.abs(shap_values[i])
            
            # Normalize to percentages (sum to 100%)
            total_shap = np.sum(sample_shap)
            if total_shap > 0:
                normalized_shap = (sample_shap / total_shap) * 100
            else:
                normalized_shap = np.zeros_like(sample_shap)
            
            sample_contributions = []
            for idx, feature_name in enumerate(self.feature_names):
                sample_contributions.append({
                    'feature': feature_name,
                    'contribution': float(normalized_shap[idx]),
                    'value': float(X[i, idx]),
                    'shap_value': float(shap_values[i, idx]) # Keep raw for debug
                })
            
            # Sort by contribution descending
            sample_contributions.sort(key=lambda x: x['contribution'], reverse=True)
            
            # Return top N
            contributions.append(sample_contributions[:top_n])
        
        return contributions if len(contributions) > 1 else contributions[0]
    
    def save_model(self, path=None):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if path is None:
            path = config.MODELS_DIR / "isolation_forest.pkl"
        
        model_data = {
            'model': self.model,
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'params': self.params
        }
        
        joblib.dump(model_data, path)
        print(f"   ✓ Model saved to: {path}")
    
    @staticmethod
    def load_model(path):
        """
        Load a saved model
        
        Args:
            path: Path to the saved model
            
        Returns:
            IsolationForestDetector instance
        """
        model_data = joblib.load(path)
        
        detector = IsolationForestDetector()
        detector.model = model_data['model']
        detector.explainer = model_data.get('explainer')
        detector.feature_names = model_data['feature_names']
        detector.min_score = model_data.get('min_score', -0.5) # Fallback
        detector.max_score = model_data.get('max_score', 0.5)  # Fallback
        detector.params = model_data['params']
        
        return detector


    def visualize_anomalies(self, X, timestamps, scores, labels, save_path=None):
        """
        Visualize anomalies
        
        Args:
            X: Input data
            timestamps: Timestamps corresponding to data
            scores: Anomaly scores
            labels: Binary labels
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Anomaly scores over time
        ax1 = fig.add_subplot(gs[0, :])
        
        # Get colors based on severity
        # We need to re-categorize to get labels for coloring if not provided
        if isinstance(scores[0], (int, float, np.number)):
            severity_info = self.categorize_severity(scores)
            severity_labels = [s['severity_label'].lower().split()[0] for s in severity_info]
        else:
            severity_labels = ['normal'] * len(scores)
            
        colors = [config.SEVERITY_COLORS.get(sev, '#6c757d') for sev in severity_labels]
        
        ax1.scatter(timestamps, scores, c=colors, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Anomaly Score (Normalized)', fontsize=12, fontweight='bold')
        ax1.set_title('Isolation Forest: Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=config.SEVERITY_COLORS[sev], label=sev.capitalize()) 
                          for sev in ['normal', 'mild', 'moderate', 'severe']]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Score distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Plot 3: Anomaly count over time
        ax3 = fig.add_subplot(gs[1, 1])
        anomaly_timestamps = timestamps[labels == 1]
        ax3.hist(timestamps, bins=50, alpha=0.5, color='blue', label='All Samples', edgecolor='black')
        if len(anomaly_timestamps) > 0:
            ax3.hist(anomaly_timestamps, bins=50, alpha=0.7, color='red', label='Anomalies', edgecolor='black')
        ax3.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax3.set_title('Anomaly Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            print(f"   ✓ Visualization saved to: {save_path}")
        
        plt.close()

if __name__ == "__main__":
    print("Isolation Forest Detector module loaded successfully!")
