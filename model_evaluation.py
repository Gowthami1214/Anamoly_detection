"""
Model Evaluation and Comparison Module
Includes adaptive threshold recalibration for model drift
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    silhouette_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))
import config

def evaluate_model(model_name, scores, labels, severity, y_true=None):
    """
    Comprehensive evaluation of an anomaly detection model
    
    Args:
        model_name: Name of the model
        scores: Anomaly scores
        labels: Predicted labels (0=normal, 1=anomaly)
        severity: Severity categories
        y_true: Ground truth labels (optional, for supervised metrics)
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    metrics = {
        'model_name': model_name,
        'n_samples': len(scores),
        'n_anomalies_detected': np.sum(labels),
        'anomaly_rate': np.mean(labels) * 100
    }
    
    # Score statistics
    print(f"\n[1/5] Anomaly Score Statistics:")
    metrics['score_mean'] = np.mean(scores)
    metrics['score_std'] = np.std(scores)
    metrics['score_min'] = np.min(scores)
    metrics['score_max'] = np.max(scores)
    metrics['score_median'] = np.median(scores)
    
    print(f"   • Mean: {metrics['score_mean']:.4f}")
    print(f"   • Std: {metrics['score_std']:.4f}")
    print(f"   • Min: {metrics['score_min']:.4f}")
    print(f"   • Max: {metrics['score_max']:.4f}")
    print(f"   • Median: {metrics['score_median']:.4f}")
    
    # Severity distribution
    print(f"\n[2/5] Severity Distribution:")
    severity_counts = pd.Series(severity).value_counts()
    for sev in ['normal', 'mild', 'moderate', 'severe']:
        count = severity_counts.get(sev, 0)
        pct = (count / len(severity)) * 100
        metrics[f'severity_{sev}_count'] = count
        metrics[f'severity_{sev}_pct'] = pct
        print(f"   • {sev.capitalize()}: {count} ({pct:.2f}%)")
    
    # Silhouette score (clustering quality)
    print(f"\n[3/5] Clustering Quality:")
    try:
        # Create feature matrix from scores for silhouette
        X_scores = scores.reshape(-1, 1)
        sil_score = silhouette_score(X_scores, labels)
        metrics['silhouette_score'] = sil_score
        print(f"   • Silhouette Score: {sil_score:.4f}")
    except:
        metrics['silhouette_score'] = None
        print(f"   • Silhouette Score: N/A")
    
    # Supervised metrics (if ground truth available)
    if y_true is not None:
        print(f"\n[4/5] Supervised Metrics (using ground truth):")
        
        # Convert ground truth to binary (0=normal, >0=anomaly)
        y_true_binary = (y_true > 0).astype(int)
        
        metrics['precision'] = precision_score(y_true_binary, labels, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, labels, zero_division=0)
        metrics['f1_score'] = f1_score(y_true_binary, labels, zero_division=0)
        
        print(f"   • Precision: {metrics['precision']:.4f}")
        print(f"   • Recall: {metrics['recall']:.4f}")
        print(f"   • F1-Score: {metrics['f1_score']:.4f}")
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, scores)
            print(f"   • ROC-AUC: {metrics['roc_auc']:.4f}")
        except:
            metrics['roc_auc'] = None
            print(f"   • ROC-AUC: N/A")
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, labels)
        metrics['confusion_matrix'] = cm
        print(f"\n   Confusion Matrix:")
        print(f"   {cm}")
    else:
        print(f"\n[4/5] Supervised Metrics: N/A (no ground truth)")
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1_score'] = None
        metrics['roc_auc'] = None
    
    print(f"\n[5/5] Summary:")
    print(f"   • Total Samples: {metrics['n_samples']}")
    print(f"   • Anomalies Detected: {metrics['n_anomalies_detected']} ({metrics['anomaly_rate']:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION COMPLETE")
    print(f"{'='*60}\n")
    
    return metrics


def compare_models(model_results, save_path=None):
    """
    Compare multiple models and generate comparison visualizations
    
    Args:
        model_results: Dictionary of {model_name: metrics_dict}
        save_path: Path to save comparison plot
        
    Returns:
        Comparison DataFrame
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Anomaly Rate (%)': metrics['anomaly_rate'],
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1-Score': metrics.get('f1_score', np.nan),
            'ROC-AUC': metrics.get('roc_auc', np.nan),
            'Silhouette': metrics.get('silhouette_score', np.nan),
            'Severe (%)': metrics.get('severity_severe_pct', np.nan)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("Comparison Table:")
    print(df_comparison.to_string(index=False))
    print()
    
    # Visualize comparison
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: F1-Score comparison
        ax = axes[0, 0]
        valid_f1 = df_comparison.dropna(subset=['F1-Score'])
        if not valid_f1.empty:
            bars = ax.bar(valid_f1['Model'], valid_f1['F1-Score'], color='steelblue', edgecolor='black')
            ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
            ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No ground truth available', ha='center', va='center')
            ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        
        # Plot 2: ROC-AUC comparison
        ax = axes[0, 1]
        valid_auc = df_comparison.dropna(subset=['ROC-AUC'])
        if not valid_auc.empty:
            bars = ax.bar(valid_auc['Model'], valid_auc['ROC-AUC'], color='coral', edgecolor='black')
            ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
            ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No ground truth available', ha='center', va='center')
            ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        
        # Plot 3: Anomaly detection rate
        ax = axes[1, 0]
        bars = ax.bar(df_comparison['Model'], df_comparison['Anomaly Rate (%)'], 
                     color='lightgreen', edgecolor='black')
        ax.set_ylabel('Anomaly Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Anomaly Detection Rate', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Severity distribution
        ax = axes[1, 1]
        severity_data = df_comparison[['Model', 'Severe (%)']].dropna()
        if not severity_data.empty:
            bars = ax.bar(severity_data['Model'], severity_data['Severe (%)'], 
                         color='crimson', edgecolor='black')
            ax.set_ylabel('Severe Anomalies (%)', fontsize=12, fontweight='bold')
            ax.set_title('Severe Anomaly Rate', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"✓ Comparison visualization saved to: {save_path}\n")
        plt.close()
    
    print(f"{'='*60}")
    print("MODEL COMPARISON COMPLETE")
    print(f"{'='*60}\n")
    
    return df_comparison


def select_best_model(model_results):
    """
    Select the best model based on F1-score and ROC-AUC
    
    Args:
        model_results: Dictionary of {model_name: metrics_dict}
        
    Returns:
        best_model_name: Name of the best model
    """
    print(f"\n{'='*60}")
    print("SELECTING BEST MODEL")
    print(f"{'='*60}\n")
    
    # Calculate composite score (F1 + AUC) / 2
    scores = {}
    for model_name, metrics in model_results.items():
        f1 = metrics.get('f1_score', 0) or 0
        auc = metrics.get('roc_auc', 0) or 0
        
        if f1 > 0 and auc > 0:
            composite_score = (f1 + auc) / 2
        elif f1 > 0:
            composite_score = f1
        elif auc > 0:
            composite_score = auc
        else:
            # If no ground truth, use silhouette score
            composite_score = metrics.get('silhouette_score', 0) or 0
        
        scores[model_name] = composite_score
        print(f"{model_name}:")
        print(f"   • F1-Score: {f1:.4f}")
        print(f"   • ROC-AUC: {auc:.4f}")
        print(f"   • Composite Score: {composite_score:.4f}\n")
    
    best_model = max(scores, key=scores.get)
    
    print(f"{'='*60}")
    print(f"BEST MODEL: {best_model.upper()}")
    print(f"Score: {scores[best_model]:.4f}")
    print(f"{'='*60}\n")
    
    return best_model


def update_threshold(model, new_data, percentile=95):
    """
    Adaptive threshold recalibration for model drift
    
    Args:
        model: Trained anomaly detection model
        new_data: New data for threshold recalibration
        percentile: Percentile for threshold (default: 95)
        
    Returns:
        new_threshold: Recalibrated threshold
        drift_detected: Boolean indicating if significant drift detected
    """
    print(f"\n{'='*60}")
    print("ADAPTIVE THRESHOLD RECALIBRATION")
    print(f"{'='*60}\n")
    
    print(f"[1/3] Computing anomaly scores on new data...")
    print(f"   • New data samples: {len(new_data)}")
    
    # Get scores from model (method depends on model type)
    if hasattr(model, 'predict'):
        if hasattr(model, 'model') and hasattr(model.model, 'decision_function'):
            # Isolation Forest
            scores = model.model.decision_function(new_data)
            scores = -scores  # Invert for consistency
        else:
            # Neural network models
            scores, _, _ = model.predict(new_data)
    else:
        raise ValueError("Model must have a predict method")
    
    print(f"   ✓ Scores computed")
    
    # Calculate new threshold
    print(f"\n[2/3] Calculating new threshold...")
    new_threshold = np.percentile(scores, percentile)
    
    # Get old threshold
    if hasattr(model, 'severity_thresholds'):
        old_threshold = model.severity_thresholds.get('mild', new_threshold)
    else:
        old_threshold = new_threshold
    
    print(f"   • Old threshold: {old_threshold:.6f}")
    print(f"   • New threshold: {new_threshold:.6f}")
    
    # Detect drift
    print(f"\n[3/3] Checking for drift...")
    threshold_change = abs(new_threshold - old_threshold) / old_threshold * 100
    drift_detected = threshold_change > 10  # 10% change threshold
    
    print(f"   • Threshold change: {threshold_change:.2f}%")
    print(f"   • Drift detected: {'YES' if drift_detected else 'NO'}")
    
    if drift_detected:
        print(f"\n   ⚠ Significant drift detected! Consider retraining the model.")
        print(f"   → Updating threshold to: {new_threshold:.6f}")
        
        # Update model threshold
        if hasattr(model, 'severity_thresholds'):
            model.severity_thresholds['mild'] = new_threshold
    else:
        print(f"\n   ✓ No significant drift detected. Threshold stable.")
    
    print(f"\n{'='*60}")
    print("THRESHOLD RECALIBRATION COMPLETE")
    print(f"{'='*60}\n")
    
    return new_threshold, drift_detected


def calculate_severity_thresholds(scores, method='percentile'):
    """
    Calculate optimal severity thresholds
    
    Args:
        scores: Anomaly scores
        method: Method for threshold calculation ('percentile' or 'kmeans')
        
    Returns:
        Dictionary of severity thresholds
    """
    if method == 'percentile':
        thresholds = {
            'normal': np.percentile(scores, 50),
            'mild': np.percentile(scores, 70),
            'moderate': np.percentile(scores, 90),
            'severe': np.percentile(scores, 95)
        }
    else:
        # Could implement k-means clustering here
        thresholds = {
            'normal': np.percentile(scores, 50),
            'mild': np.percentile(scores, 70),
            'moderate': np.percentile(scores, 90),
            'severe': np.percentile(scores, 95)
        }
    
    return thresholds


if __name__ == "__main__":
    print("Model Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  • evaluate_model(model_name, scores, labels, severity, y_true)")
    print("  • compare_models(model_results, save_path)")
    print("  • select_best_model(model_results)")
    print("  • update_threshold(model, new_data, percentile)")
    print("  • calculate_severity_thresholds(scores, method)")
