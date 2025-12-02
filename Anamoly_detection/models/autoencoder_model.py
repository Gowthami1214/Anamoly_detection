"""
Autoencoder Anomaly Detection Model with Per-Feature Explainability
Implements fully connected autoencoder for anomaly detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    """
    Fully connected Autoencoder architecture
    """
    
    def __init__(self, input_dim, encoding_dims, dropout_rate=0.2):
        """
        Args:
            input_dim: Number of input features
            encoding_dims: List of encoder layer dimensions [64, 32, 16]
            dropout_rate: Dropout rate for regularization
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        decoding_dims = encoding_dims[::-1][1:] + [input_dim]
        
        for dim in decoding_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Identity(),
                nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity(),
                nn.Dropout(dropout_rate) if dim != input_dim else nn.Identity()
            ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector with per-feature explainability
    """
    
    def __init__(self, **params):
        """
        Initialize Autoencoder detector
        
        Args:
            **params: Model parameters (encoding_dims, learning_rate, etc.)
        """
        self.params = {**config.AUTOENCODER_DEFAULT, **params}
        self.model = None
        self.feature_names = None
        self.severity_thresholds = None
        self.train_losses = []
        self.val_losses = []
    
    def build_model(self, input_dim):
        """
        Build the autoencoder model
        
        Args:
            input_dim: Number of input features
        """
        self.model = Autoencoder(
            input_dim=input_dim,
            encoding_dims=self.params['encoding_dims'],
            dropout_rate=self.params['dropout_rate']
        ).to(device)
        
        print(f"\n   Model Architecture:")
        print(f"   • Input dimension: {input_dim}")
        print(f"   • Encoding dimensions: {self.params['encoding_dims']}")
        print(f"   • Dropout rate: {self.params['dropout_rate']}")
        print(f"   • Device: {device}")
    
    def train(self, X_train, feature_names=None, X_val=None):
        """
        Train the Autoencoder model
        
        Args:
            X_train: Training data
            feature_names: List of feature names
            X_val: Validation data (optional)
        """
        print("\n" + "=" * 60)
        print("TRAINING AUTOENCODER MODEL")
        print("=" * 60)
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        print(f"\n[1/4] Preparing data...")
        print(f"   • Training samples: {X_train.shape[0]}")
        print(f"   • Features: {X_train.shape[1]}")
        
        # Split validation set if not provided
        if X_val is None:
            val_size = int(len(X_train) * config.VALIDATION_SPLIT)
            X_val = X_train[-val_size:]
            X_train = X_train[:-val_size]
            print(f"   • Validation samples: {X_val.shape[0]} (split from training)")
        else:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            print(f"   • Validation samples: {X_val.shape[0]}")
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(X_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )
        
        # Build model
        print(f"\n[2/4] Building model...")
        self.build_model(X_train.shape[1])
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['learning_rate']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        print(f"\n[3/4] Training model...")
        print(f"   • Epochs: {self.params['epochs']}")
        print(f"   • Batch size: {self.params['batch_size']}")
        print(f"   • Learning rate: {self.params['learning_rate']}")
        print(f"   • Early stopping patience: {self.params['patience']}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.params['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch [{epoch+1}/{self.params['epochs']}] - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.params['patience']:
                    print(f"\n   ⚠ Early stopping triggered at epoch {epoch+1}")
                    self.model.load_state_dict(best_model_state)
                    break
        
        print(f"   ✓ Training complete - Best Val Loss: {best_val_loss:.6f}")
        
        # Calculate severity thresholds
        print(f"\n[4/4] Calculating severity thresholds...")
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            reconstructed = self.model(X_train_tensor).cpu().numpy()
            train_errors = np.mean((X_train - reconstructed) ** 2, axis=1)
        
        self.severity_thresholds = {
            'normal': np.percentile(train_errors, 50),
            'mild': np.percentile(train_errors, 70),
            'moderate': np.percentile(train_errors, 90),
            'severe': np.percentile(train_errors, 95)
        }
        print(f"   ✓ Severity thresholds calculated:")
        for severity, threshold in self.severity_thresholds.items():
            print(f"     - {severity.capitalize()}: {threshold:.6f}")
        
        print("\n" + "=" * 60)
        print("AUTOENCODER TRAINING COMPLETE")
        print("=" * 60 + "\n")
    
    def predict(self, X):
        """
        Predict anomaly scores using reconstruction error
        
        Args:
            X: Input data
            
        Returns:
            scores: Anomaly scores (reconstruction errors)
            labels: Binary labels (0=normal, 1=anomaly)
            severity: Severity categories
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            reconstructed = self.model(X_tensor).cpu().numpy()
        
        # Calculate reconstruction error per sample
        scores = np.mean((X - reconstructed) ** 2, axis=1)
        
        # Normalize scores to 0-1 range
        scores_normalized = self._normalize_scores(scores)
        
        # Determine labels (using mild threshold)
        labels = (scores > self.severity_thresholds['mild']).astype(int)
        
        # Categorize severity
        severity = self.categorize_severity(scores)
        
        return scores_normalized, labels, severity
    
    def _normalize_scores(self, scores):
        """Normalize scores to 0-1 range"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def categorize_severity(self, scores):
        """Categorize reconstruction errors into severity levels"""
        severity = np.empty(len(scores), dtype=object)
        
        for i, score in enumerate(scores):
            if score < self.severity_thresholds['normal']:
                severity[i] = 'normal'
            elif score < self.severity_thresholds['mild']:
                severity[i] = 'mild'
            elif score < self.severity_thresholds['moderate']:
                severity[i] = 'moderate'
            else:
                severity[i] = 'severe'
        
        return severity
    
    def get_feature_contributions(self, X, top_n=5):
        """
        Get per-feature reconstruction errors for explainability
        
        Args:
            X: Input data (single sample or batch)
            top_n: Number of top features to return
            
        Returns:
            List of dictionaries with feature contributions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            reconstructed = self.model(X_tensor).cpu().numpy()
        
        # Calculate per-feature reconstruction errors
        feature_errors = (X - reconstructed) ** 2
        
        contributions = []
        for i in range(len(X)):
            # Get top N features with highest reconstruction error
            top_indices = np.argsort(feature_errors[i])[-top_n:][::-1]
            
            sample_contributions = []
            for idx in top_indices:
                sample_contributions.append({
                    'feature': self.feature_names[idx],
                    'contribution': feature_errors[i, idx],
                    'value': X[i, idx],
                    'reconstructed': reconstructed[i, idx],
                    'error': feature_errors[i, idx]
                })
            
            contributions.append(sample_contributions)
        
        return contributions if len(contributions) > 1 else contributions[0]
    
    def visualize_anomalies(self, X, timestamps, scores, labels, save_path=None):
        """Visualize anomalies and training history"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training history
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.train_losses, label='Training Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax1.set_title('Autoencoder Training History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Anomaly scores over time
        ax2 = fig.add_subplot(gs[1, :])
        colors = [config.SEVERITY_COLORS[sev] for sev in self.categorize_severity(scores)]
        ax2.scatter(timestamps, scores, c=colors, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax2.set_title('Autoencoder: Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=config.SEVERITY_COLORS[sev], label=sev.capitalize()) 
                          for sev in ['normal', 'mild', 'moderate', 'severe']]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Plot 3: Score distribution
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Anomaly count over time
        ax4 = fig.add_subplot(gs[2, 1])
        anomaly_timestamps = timestamps[labels == 1]
        ax4.hist(timestamps, bins=50, alpha=0.5, color='blue', label='All Samples', edgecolor='black')
        ax4.hist(anomaly_timestamps, bins=50, alpha=0.7, color='red', label='Anomalies', edgecolor='black')
        ax4.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Anomaly Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            print(f"   ✓ Visualization saved to: {save_path}")
        
        plt.close()
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = config.MODELS_DIR / "autoencoder.pkl"
        
        model_data = {
            'model_state': self.model.state_dict(),
            'params': self.params,
            'feature_names': self.feature_names,
            'severity_thresholds': self.severity_thresholds,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(model_data, path)
        print(f"   ✓ Model saved to: {path}")
    
    @staticmethod
    def load_model(path):
        """Load a saved model"""
        model_data = torch.load(path, map_location=device)
        
        detector = AutoencoderDetector(**model_data['params'])
        
        # Rebuild model architecture
        input_dim = len(model_data['feature_names'])
        detector.build_model(input_dim)
        detector.model.load_state_dict(model_data['model_state'])
        detector.model.eval()
        
        detector.feature_names = model_data['feature_names']
        detector.severity_thresholds = model_data['severity_thresholds']
        detector.train_losses = model_data['train_losses']
        detector.val_losses = model_data['val_losses']
        
        return detector


if __name__ == "__main__":
    print("Autoencoder Detector module loaded successfully!")
