"""
LSTM Autoencoder Anomaly Detection Model with Time-Step Explainability
Implements sequence-based anomaly detection for time-series data
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

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection
    """
    
    def __init__(self, input_dim, lstm_units, num_layers, dropout_rate=0.2):
        """
        Args:
            input_dim: Number of features
            lstm_units: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(lstm_units, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Reconstructed sequence
        """
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use last encoded state as input to decoder
        # Repeat for each time step
        batch_size, seq_len, _ = x.shape
        decoder_input = encoded[:, -1:, :].repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        decoded = self.dropout(decoded)
        
        # Output
        output = self.output_layer(decoded)
        
        return output


class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder-based anomaly detector with time-step explainability
    """
    
    def __init__(self, **params):
        """
        Initialize LSTM Autoencoder detector
        
        Args:
            **params: Model parameters (lstm_units, sequence_length, etc.)
        """
        self.params = {**config.LSTM_AUTOENCODER_DEFAULT, **params}
        self.model = None
        self.feature_names = None
        self.severity_thresholds = None
        self.train_losses = []
        self.val_losses = []
    
    def build_model(self, input_dim):
        """
        Build the LSTM autoencoder model
        
        Args:
            input_dim: Number of input features
        """
        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            lstm_units=self.params['lstm_units'],
            num_layers=self.params['num_layers'],
            dropout_rate=self.params['dropout_rate']
        ).to(device)
        
        print(f"\n   Model Architecture:")
        print(f"   • Input dimension: {input_dim}")
        print(f"   • LSTM units: {self.params['lstm_units']}")
        print(f"   • Number of layers: {self.params['num_layers']}")
        print(f"   • Sequence length: {self.params['sequence_length']}")
        print(f"   • Dropout rate: {self.params['dropout_rate']}")
        print(f"   • Device: {device}")
    
    def create_sequences(self, X, timestamps=None):
        """
        Create sequences from time-series data using sliding window
        
        Args:
            X: Input data (numpy array or DataFrame)
            timestamps: Timestamp array (optional)
            
        Returns:
            sequences: 3D array of shape (n_sequences, sequence_length, n_features)
            sequence_timestamps: Timestamps for each sequence (if provided)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        seq_len = self.params['sequence_length']
        n_samples, n_features = X.shape
        
        # Calculate number of sequences
        n_sequences = n_samples - seq_len + 1
        
        # Create sequences
        sequences = np.zeros((n_sequences, seq_len, n_features))
        for i in range(n_sequences):
            sequences[i] = X[i:i+seq_len]
        
        # Create sequence timestamps (use last timestamp of each sequence)
        if timestamps is not None:
            if isinstance(timestamps, pd.Series):
                timestamps = timestamps.values
            sequence_timestamps = timestamps[seq_len-1:]
        else:
            sequence_timestamps = None
        
        return sequences, sequence_timestamps
    
    def train(self, X_train, feature_names=None, timestamps=None, X_val=None):
        """
        Train the LSTM Autoencoder model
        
        Args:
            X_train: Training data
            feature_names: List of feature names
            timestamps: Timestamps for creating sequences
            X_val: Validation data (optional)
        """
        print("\n" + "=" * 60)
        print("TRAINING LSTM AUTOENCODER MODEL")
        print("=" * 60)
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        print(f"\n[1/5] Preparing data...")
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
        
        print(f"\n[2/5] Creating sequences...")
        print(f"   • Sequence length: {self.params['sequence_length']}")
        
        # Create sequences
        train_sequences, _ = self.create_sequences(X_train)
        val_sequences, _ = self.create_sequences(X_val)
        
        print(f"   • Training sequences: {train_sequences.shape[0]}")
        print(f"   • Validation sequences: {val_sequences.shape[0]}")
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_sequences)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_sequences)
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
        print(f"\n[3/5] Building model...")
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
        print(f"\n[4/5] Training model...")
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
        print(f"\n[5/5] Calculating severity thresholds...")
        self.model.eval()
        with torch.no_grad():
            train_tensor = torch.FloatTensor(train_sequences).to(device)
            reconstructed = self.model(train_tensor).cpu().numpy()
            train_errors = np.mean((train_sequences - reconstructed) ** 2, axis=(1, 2))
        
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
        print("LSTM AUTOENCODER TRAINING COMPLETE")
        print("=" * 60 + "\n")
    
    def predict(self, X, timestamps=None):
        """
        Predict anomaly scores using sequence reconstruction error
        
        Args:
            X: Input data
            timestamps: Timestamps (optional)
            
        Returns:
            scores: Anomaly scores
            labels: Binary labels
            severity: Severity categories
            sequence_timestamps: Timestamps for sequences
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create sequences
        sequences, sequence_timestamps = self.create_sequences(X, timestamps)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sequences).to(device)
            reconstructed = self.model(X_tensor).cpu().numpy()
        
        # Calculate reconstruction error per sequence
        scores = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        
        # Normalize scores to 0-1 range
        scores_normalized = self._normalize_scores(scores)
        
        # Determine labels
        labels = (scores > self.severity_thresholds['mild']).astype(int)
        
        # Categorize severity
        severity = self.categorize_severity(scores)
        
        return scores_normalized, labels, severity, sequence_timestamps
    
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
    
    def get_timestep_errors(self, X, top_n=5):
        """
        Get time-step error heatmaps for explainability
        
        Args:
            X: Input data (single sequence or batch)
            top_n: Number of top features to return per timestep
            
        Returns:
            Dictionary with timestep errors and feature contributions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create sequences
        sequences, _ = self.create_sequences(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sequences).to(device)
            reconstructed = self.model(X_tensor).cpu().numpy()
        
        # Calculate per-timestep, per-feature errors
        timestep_errors = (sequences - reconstructed) ** 2
        
        # For each sequence, get top contributing features at each timestep
        contributions = []
        for seq_idx in range(len(sequences)):
            seq_contributions = []
            for t in range(self.params['sequence_length']):
                # Get top features for this timestep
                top_indices = np.argsort(timestep_errors[seq_idx, t])[-top_n:][::-1]
                
                timestep_contrib = {
                    'timestep': t,
                    'features': []
                }
                
                for feat_idx in top_indices:
                    timestep_contrib['features'].append({
                        'feature': self.feature_names[feat_idx],
                        'error': timestep_errors[seq_idx, t, feat_idx],
                        'original': sequences[seq_idx, t, feat_idx],
                        'reconstructed': reconstructed[seq_idx, t, feat_idx]
                    })
                
                seq_contributions.append(timestep_contrib)
            
            contributions.append(seq_contributions)
        
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
        ax1.set_title('LSTM Autoencoder Training History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Anomaly scores over time
        ax2 = fig.add_subplot(gs[1, :])
        colors = [config.SEVERITY_COLORS[sev] for sev in self.categorize_severity(scores)]
        ax2.scatter(timestamps, scores, c=colors, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sequence Reconstruction Error', fontsize=12, fontweight='bold')
        ax2.set_title('LSTM Autoencoder: Anomaly Scores Over Time', fontsize=14, fontweight='bold')
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
        ax3.set_xlabel('Sequence Reconstruction Error', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Anomaly count over time
        ax4 = fig.add_subplot(gs[2, 1])
        anomaly_timestamps = timestamps[labels == 1]
        ax4.hist(timestamps, bins=50, alpha=0.5, color='blue', label='All Sequences', edgecolor='black')
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
            path = config.MODELS_DIR / "lstm_autoencoder.pkl"
        
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
        
        detector = LSTMAutoencoderDetector(**model_data['params'])
        
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
    print("LSTM Autoencoder Detector module loaded successfully!")
