#!/usr/bin/env python3
"""
Neural Network Training for Confusion Detection
Trains models on EEG/fNIRS data to detect confusion in real-time
Supports loading multiple NPZ files and various model architectures
"""

import numpy as np
import os
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ConfusionDataset:
    """Dataset class for loading and preprocessing confusion detection data"""
    
    def __init__(self, npz_files, window_size=256, stride=64, balance_classes=True):
        """
        Args:
            npz_files: List of NPZ file paths
            window_size: Number of samples per window (256 = 1 second at 256Hz)
            stride: Stride for sliding window
            balance_classes: Whether to balance positive/negative samples
        """
        self.window_size = window_size
        self.stride = stride
        self.balance_classes = balance_classes
        
        # Load all data
        self.all_data = []
        self.all_events = []
        self.all_metadata = []
        
        print(f"Loading {len(npz_files)} NPZ files...")
        for file_path in npz_files:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract data
            timestamps = data['timestamps']
            eeg = data['eeg'] if 'eeg' in data else None
            fnirs = data['fnirs'] if 'fnirs' in data else None
            words = data['words'] if 'words' in data else None
            
            # Extract events
            event_timestamps = data['event_timestamps'] if 'event_timestamps' in data else []
            event_types = data['event_types'] if 'event_types' in data else []
            event_words = data['event_words'] if 'event_words' in data else []
            
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            print(f"  Loaded: {os.path.basename(file_path)}")
            print(f"    Samples: {len(timestamps)}")
            print(f"    Events: {len(event_timestamps)}")
            
            self.all_data.append({
                'timestamps': timestamps,
                'eeg': eeg,
                'fnirs': fnirs,
                'words': words,
                'event_timestamps': event_timestamps,
                'event_types': event_types,
                'event_words': event_words,
                'metadata': metadata
            })
        
        # Create windows and labels
        self.windows, self.labels, self.window_info = self._create_windows()
        
        print(f"\nDataset created:")
        print(f"  Total windows: {len(self.windows)}")
        print(f"  Confusion windows: {np.sum(self.labels == 1)}")
        print(f"  Normal windows: {np.sum(self.labels == 0)}")
        print(f"  Class balance: {np.mean(self.labels):.3f}")
    
    def _create_windows(self):
        """Create sliding windows and assign labels based on confusion events"""
        all_windows = []
        all_labels = []
        all_window_info = []
        
        for file_idx, file_data in enumerate(self.all_data):
            timestamps = file_data['timestamps']
            eeg = file_data['eeg']
            fnirs = file_data['fnirs']
            
            if eeg is None or len(eeg) == 0:
                continue
            
            # Convert to numpy arrays if needed
            eeg = np.array(eeg) if not isinstance(eeg, np.ndarray) else eeg
            fnirs = np.array(fnirs) if not isinstance(fnirs, np.ndarray) else fnirs
            
            # Get confusion event times
            confusion_times = []
            for t, event_type in zip(file_data['event_timestamps'], file_data['event_types']):
                if 'confusion' in event_type:
                    confusion_times.append(t)
            
            # Create sliding windows
            for start_idx in range(0, len(timestamps) - self.window_size, self.stride):
                end_idx = start_idx + self.window_size
                
                # Get window timestamps
                window_timestamps = timestamps[start_idx:end_idx]
                window_start_time = window_timestamps[0]
                window_end_time = window_timestamps[-1]
                
                # Check if any confusion event falls within this window
                # Also check slightly before the window (reaction time)
                label = 0
                for conf_time in confusion_times:
                    if window_start_time - 0.5 <= conf_time <= window_end_time:
                        label = 1
                        break
                
                # Extract window data
                window_eeg = eeg[start_idx:end_idx]
                window_fnirs = fnirs[start_idx:end_idx] if fnirs is not None else None
                
                # Skip if data is invalid
                if len(window_eeg) != self.window_size:
                    continue
                
                all_windows.append({
                    'eeg': window_eeg,
                    'fnirs': window_fnirs,
                    'file_idx': file_idx,
                    'start_idx': start_idx
                })
                all_labels.append(label)
                all_window_info.append({
                    'file_idx': file_idx,
                    'start_time': window_start_time,
                    'end_time': window_end_time
                })
        
        # Balance classes if requested
        if self.balance_classes:
            all_windows, all_labels, all_window_info = self._balance_classes(
                all_windows, all_labels, all_window_info
            )
        
        return all_windows, np.array(all_labels), all_window_info
    
    def _balance_classes(self, windows, labels, window_info):
        """Balance positive and negative samples"""
        labels = np.array(labels)
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
        # Undersample majority class
        n_samples = min(len(pos_indices), len(neg_indices))
        
        if n_samples == 0:
            print("Warning: No positive samples found!")
            return windows, labels, window_info
        
        # Randomly sample from majority class
        if len(pos_indices) > len(neg_indices):
            selected_pos = np.random.choice(pos_indices, n_samples, replace=False)
            selected_indices = np.concatenate([selected_pos, neg_indices])
        else:
            selected_neg = np.random.choice(neg_indices, n_samples, replace=False)
            selected_indices = np.concatenate([pos_indices, selected_neg])
        
        # Shuffle
        np.random.shuffle(selected_indices)
        
        balanced_windows = [windows[i] for i in selected_indices]
        balanced_labels = labels[selected_indices]
        balanced_info = [window_info[i] for i in selected_indices]
        
        return balanced_windows, balanced_labels, balanced_info
    
    def extract_features(self, window_data):
        """Extract features from a window of data"""
        features = []
        
        eeg = window_data['eeg']
        fnirs = window_data['fnirs']
        
        # EEG features
        for ch_idx in range(4):  # 4 EEG channels
            ch_data = eeg[:, ch_idx]
            
            # Time domain features
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.min(ch_data),
                np.max(ch_data),
                skew(ch_data),
                kurtosis(ch_data)
            ])
            
            # Frequency domain features
            freqs, psd = signal.welch(ch_data, fs=256, nperseg=64)
            
            # Band powers
            delta_power = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            beta_power = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            gamma_power = np.sum(psd[(freqs >= 30) & (freqs < 50)])
            
            total_power = np.sum(psd) + 1e-10
            
            features.extend([
                delta_power / total_power,
                theta_power / total_power,
                alpha_power / total_power,
                beta_power / total_power,
                gamma_power / total_power
            ])
        
        # fNIRS features (if available)
        if fnirs is not None:
            for ch_idx in range(4):  # 4 normalized channels
                ch_data = fnirs[:, ch_idx]
                features.extend([
                    np.mean(ch_data),
                    np.std(ch_data),
                    np.max(ch_data) - np.min(ch_data),  # Range
                    np.mean(np.diff(ch_data))  # Mean derivative
                ])
        else:
            # Add zeros if no fNIRS data
            features.extend([0] * 16)
        
        return np.array(features, dtype=np.float32)


class ConfusionDetectorCNN(nn.Module):
    """CNN model for confusion detection on raw signals"""
    
    def __init__(self, n_channels=4, window_size=256, n_classes=2):
        super().__init__()
        
        # Temporal convolutions
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate size after convolutions
        conv_out_size = window_size // 8  # 3 pooling layers
        
        self.fc1 = nn.Linear(128 * conv_out_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class ConfusionDetectorLSTM(nn.Module):
    """LSTM model for confusion detection"""
    
    def __init__(self, n_channels=4, hidden_size=64, n_layers=2, n_classes=2):
        super().__init__()
        
        self.lstm = nn.LSTM(n_channels, hidden_size, n_layers, 
                           batch_first=True, dropout=0.5)
        
        self.fc = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch, time, channels)
        _, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        
        return out


class SimpleFeatureClassifier(nn.Module):
    """Simple MLP classifier for hand-crafted features"""
    
    def __init__(self, n_features, hidden_sizes=[64, 32], n_classes=2):
        super().__init__()
        
        layers = []
        in_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, n_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ConfusionTrainer:
    """Main trainer class for confusion detection models"""
    
    def __init__(self, dataset, model_type='features', device='cpu'):
        self.dataset = dataset
        self.model_type = model_type
        self.device = torch.device(device)
        
        # Extract features if using feature-based model
        if model_type == 'features':
            print("Extracting features...")
            self.X = np.array([dataset.extract_features(w) for w in dataset.windows])
            self.y = dataset.labels
            
            # Normalize features
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            
            print(f"Feature shape: {self.X.shape}")
        else:
            # For CNN/LSTM, use raw signals
            self.X = np.array([w['eeg'] for w in dataset.windows])
            self.y = dataset.labels
        
        # Split data
        self._split_data()
        
        # Create model
        self._create_model()
    
    def _split_data(self):
        """Split data into train/val/test sets"""
        # First split: train+val vs test (80/20)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Second split: train vs val (80/20 of remaining)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Val: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")
    
    def _create_model(self):
        """Create the appropriate model based on model_type"""
        if self.model_type == 'features':
            n_features = self.X_train.shape[1]
            self.model = SimpleFeatureClassifier(n_features).to(self.device)
        elif self.model_type == 'cnn':
            self.model = ConfusionDetectorCNN(n_channels=4).to(self.device)
        elif self.model_type == 'lstm':
            self.model = ConfusionDetectorLSTM(n_channels=4).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\nModel architecture ({self.model_type}):")
        print(self.model)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {n_params:,}")
    
    def train(self, epochs=50, batch_size=32, lr=0.001):
        """Train the model"""
        # Convert to PyTorch tensors
        if self.model_type == 'features':
            X_train = torch.FloatTensor(self.X_train).to(self.device)
            X_val = torch.FloatTensor(self.X_val).to(self.device)
        else:
            # For CNN: (batch, channels, time)
            # For LSTM: (batch, time, channels)
            X_train = torch.FloatTensor(self.X_train).to(self.device)
            X_val = torch.FloatTensor(self.X_val).to(self.device)
            
            if self.model_type == 'cnn':
                X_train = X_train.transpose(1, 2)
                X_val = X_val.transpose(1, 2)
        
        y_train = torch.LongTensor(self.y_train).to(self.device)
        y_val = torch.LongTensor(self.y_val).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0
        best_model_state = None
        
        print("\nTraining...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_correct = (val_predicted == y_val).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(X_train)
            train_acc = train_correct / len(X_train)
            val_loss = val_loss.item()
            val_acc = val_correct / len(X_val)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        self.history = history
        
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        
        # Prepare test data
        if self.model_type == 'features':
            X_test = torch.FloatTensor(self.X_test).to(self.device)
        else:
            X_test = torch.FloatTensor(self.X_test).to(self.device)
            if self.model_type == 'cnn':
                X_test = X_test.transpose(1, 2)
        
        y_test = torch.LongTensor(self.y_test).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_test)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        # Convert to numpy
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        y_prob = probabilities[:, 1].cpu().numpy()
        
        # Calculate metrics
        accuracy = (y_pred == y_true).mean()
        
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Normal', 'Confusion']))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    
    def plot_results(self, results):
        """Plot training history and evaluation results"""
        fig = plt.figure(figsize=(15, 10))
        
        # Training history
        plt.subplot(2, 3, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_acc'], label='Train')
        plt.plot(self.history['val_acc'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training History - Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Confusion'],
                   yticklabels=['Normal', 'Confusion'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(2, 3, 4)
        plt.plot(results['fpr'], results['tpr'], 
                label=f'ROC (AUC = {results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Probability distribution
        plt.subplot(2, 3, 5)
        plt.hist(results['y_prob'][results['y_true'] == 0], bins=30, 
                alpha=0.5, label='Normal', density=True)
        plt.hist(results['y_prob'][results['y_true'] == 1], bins=30, 
                alpha=0.5, label='Confusion', density=True)
        plt.xlabel('Predicted Probability of Confusion')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance (for feature-based model)
        if self.model_type == 'features' and hasattr(self.model, 'model'):
            plt.subplot(2, 3, 6)
            
            # Get first layer weights
            first_layer = self.model.model[0]
            weights = first_layer.weight.data.cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            
            # Get top features
            n_top = 20
            top_indices = np.argsort(importance)[-n_top:]
            
            feature_names = self._get_feature_names()
            top_names = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            plt.barh(range(n_top), top_importance)
            plt.yticks(range(n_top), top_names, fontsize=8)
            plt.xlabel('Importance')
            plt.title(f'Top {n_top} Feature Importances')
            plt.tight_layout()
        
        plt.suptitle(f'Confusion Detection Results - {self.model_type.upper()} Model', 
                    fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print additional metrics
        print(f"\nModel Performance Summary:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        # Calculate inference time
        self._test_inference_time()
    
    def _get_feature_names(self):
        """Generate feature names for interpretability"""
        names = []
        
        # EEG features
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            names.extend([
                f'{ch}_mean', f'{ch}_std', f'{ch}_min', f'{ch}_max',
                f'{ch}_skew', f'{ch}_kurt',
                f'{ch}_delta', f'{ch}_theta', f'{ch}_alpha', 
                f'{ch}_beta', f'{ch}_gamma'
            ])
        
        # fNIRS features
        for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4']:
            names.extend([
                f'{ch}_fnirs_mean', f'{ch}_fnirs_std', 
                f'{ch}_fnirs_range', f'{ch}_fnirs_deriv'
            ])
        
        return names
    
    def _test_inference_time(self):
        """Test model inference time for real-time capability"""
        import time
        
        self.model.eval()
        
        # Prepare single sample
        if self.model_type == 'features':
            sample = torch.FloatTensor(self.X_test[0:1]).to(self.device)
        else:
            sample = torch.FloatTensor(self.X_test[0:1]).to(self.device)
            if self.model_type == 'cnn':
                sample = sample.transpose(1, 2)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(sample)
        
        # Time inference
        n_runs = 100
        start_time = time.time()
        
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model(sample)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_runs * 1000  # Convert to ms
        
        print(f"\nInference Time: {avg_time:.2f} ms per sample")
        print(f"Max throughput: {1000/avg_time:.0f} samples/second")
        
        if avg_time < 10:
            print("✓ Model is suitable for real-time use (<10ms latency)")
        elif avg_time < 50:
            print("⚠ Model has moderate latency (10-50ms)")
        else:
            print("✗ Model may be too slow for real-time use (>50ms)")
    
    def save_model(self, path):
        """Save trained model and preprocessing info"""
        save_dict = {
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'model_config': {
                'n_features': self.X_train.shape[1] if self.model_type == 'features' else None,
                'window_size': self.dataset.window_size,
                'stride': self.dataset.stride
            },
            'scaler': self.scaler if self.model_type == 'features' else None,
            'history': self.history
        }
        
        torch.save(save_dict, path)
        print(f"\nModel saved to: {path}")


def main():
    """Main training pipeline"""
    
    # Get list of NPZ files
    print("Looking for NPZ files...")
    npz_files = []
    
    # Check current directory
    npz_files.extend(glob.glob("*.npz"))
    
    # Check Downloads folder
    downloads_path = os.path.expanduser("~/Downloads")
    if os.path.exists(downloads_path):
        npz_files.extend(glob.glob(os.path.join(downloads_path, "muse_athena_*.npz")))
    
    # Remove duplicates
    npz_files = list(set(npz_files))
    
    if len(npz_files) == 0:
        print("No NPZ files found! Please ensure you have recorded data using eegtrainer.py")
        return
    
    print(f"\nFound {len(npz_files)} NPZ files:")
    for f in sorted(npz_files):
        print(f"  - {os.path.basename(f)}")
    
    # Load dataset
    dataset = ConfusionDataset(
        npz_files=sorted(npz_files),
        window_size=256,  # 1 second at 256Hz
        stride=64,        # 250ms stride
        balance_classes=True
    )
    
    if len(dataset.windows) == 0:
        print("\nNo valid windows found in the data!")
        return
    
    # Try different model types
    model_types = ['features', 'cnn', 'lstm']
    results_summary = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*60}")
        
        # Create trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        trainer = ConfusionTrainer(dataset, model_type=model_type, device=device)
        
        # Train model
        history = trainer.train(epochs=50, batch_size=32, lr=0.001)
        
        # Evaluate
        results = trainer.evaluate()
        results_summary[model_type] = results
        
        # Plot results
        trainer.plot_results(results)
        
        # Save model
        model_path = f"confusion_detector_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        trainer.save_model(model_path)
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<10} {'ROC AUC':<10} {'Best For':<30}")
    print("-"*65)
    
    recommendations = {
        'features': 'Fast inference, interpretable',
        'cnn': 'Raw signal patterns',
        'lstm': 'Temporal dependencies'
    }
    
    for model_type in model_types:
        if model_type in results_summary:
            acc = results_summary[model_type]['accuracy']
            auc_score = results_summary[model_type]['roc_auc']
            print(f"{model_type.upper():<15} {acc:<10.4f} {auc_score:<10.4f} {recommendations[model_type]:<30}")
    
    print("\n✓ Training complete! Models saved for real-time use.")


if __name__ == "__main__":
    main()