#!/usr/bin/env python3
"""
=============================================================================
NEUROADAPTIVE READER - Confusion Detection RNN Training (Google Colab)
=============================================================================

This is a SINGLE FILE that contains everything needed to train the confusion
detection RNN model. Just paste this into Google Colab and run!

SETUP INSTRUCTIONS:
1. Upload your NewRecordings folder to /content/NewRecordings/
   - Use the Colab file browser or run:
     !mkdir -p /content/NewRecordings
     # Then upload your .npz files

2. Run this entire script

3. Download the trained model (confusion_rnn_best.pth) when done

The script is optimized for A100 GPU with mixed precision training.
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data settings
    'data_dir': '/content/NewRecordings',  # Colab path
    'sample_rate': 256,  # Hz
    
    # Model architecture - MUST MATCH confusion_rnn.py defaults!
    'model_type': 'lstm',  # 'lstm' or 'gru'
    'input_size': 12,      # 4 EEG + 8 fNIRS channels
    'hidden_size': 64,     # Must match confusion_rnn.py default
    'num_layers': 2,       # Must match confusion_rnn.py default
    'dropout': 0.3,
    'bidirectional': True,
    'use_attention': True,
    
    # Training settings
    'window_size_sec': 1.0,    # 1 second windows
    'step_size_sec': 0.125,    # 125ms step (8 predictions/sec)
    'batch_size': 256,         # Large batch for A100
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'patience': 15,            # Early stopping patience
    
    # A100 optimizations
    'use_amp': True,           # Mixed precision training
    'num_workers': 4,          # DataLoader workers
    'pin_memory': True,
    
    # Output
    'save_dir': '/content',
    'model_name': 'confusion_rnn_best.pth'
}

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
# NOTE: These architectures MUST match confusion_rnn.py exactly for inference!

class ConfusionDetectorRNN(nn.Module):
    """
    LSTM-based RNN for detecting confusion from EEG/fNIRS signals.
    
    Architecture:
    - Input normalization (batch norm)
    - Bidirectional LSTM layers for temporal pattern extraction
    - Attention mechanism to focus on relevant time steps
    - Classification head with dropout for regularization
    
    Designed for real-time inference with sliding windows.
    """
    
    def __init__(
        self,
        input_size: int = 12,  # 4 EEG + 8 fNIRS channels
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * self.num_directions
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Logits of shape (batch_size, 1) for binary classification
        """
        batch_size, seq_len, _ = x.shape
        
        # Input normalization: reshape for BatchNorm1d
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # (batch, seq, features)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq, hidden * num_directions)
        
        if self.use_attention:
            # Compute attention weights
            attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # Apply attention to get context vector
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden * num_directions)
        else:
            # Use last hidden state (concat forward and backward for bidirectional)
            if self.bidirectional:
                # h_n: (num_layers * num_directions, batch, hidden)
                forward_h = h_n[-2, :, :]  # Last layer forward
                backward_h = h_n[-1, :, :]  # Last layer backward
                context = torch.cat([forward_h, backward_h], dim=1)
            else:
                context = h_n[-1, :, :]
        
        # Classification
        logits = self.classifier(context)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability of confusion (0-1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


class ConfusionDetectorGRU(nn.Module):
    """
    Lighter GRU variant for faster inference.
    
    GRU has fewer parameters than LSTM while maintaining good performance
    on sequential data. Better for real-time applications.
    """
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,  # Unidirectional for lower latency
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * self.num_directions
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        # Normalize input
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        
        # GRU forward
        gru_out, h_n = self.gru(x)
        
        # Use last hidden state
        if self.bidirectional:
            context = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            context = h_n[-1]
        
        logits = self.classifier(context)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


# =============================================================================
# DATASET
# =============================================================================

class ConfusionDataset(Dataset):
    """
    PyTorch Dataset for confusion detection training.
    Creates sliding windows from EEG/fNIRS data with binary confusion labels.
    """
    
    def __init__(
        self,
        eeg: np.ndarray,
        fnirs: np.ndarray,
        labels: np.ndarray,
        window_size: int = 256,
        step_size: int = 32,
        augment: bool = False
    ):
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.fnirs = torch.tensor(fnirs, dtype=torch.float32)
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        self.augment = augment
        
        self.n_samples = len(eeg)
        self.n_windows = max(0, (self.n_samples - window_size) // step_size + 1)
        
        # Precompute window labels
        self.window_labels = self._compute_window_labels()
        
        # Precompute window indices for faster access
        self.window_starts = torch.arange(0, self.n_windows * step_size, step_size)
        
    def _compute_window_labels(self) -> torch.Tensor:
        window_labels = torch.zeros(self.n_windows, dtype=torch.float32)
        for i in range(self.n_windows):
            start = i * self.step_size
            end = start + self.window_size
            window_labels[i] = float(np.mean(self.labels[start:end]) > 0.5)
        return window_labels
    
    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int):
        start = self.window_starts[idx].item()
        end = start + self.window_size
        
        # Extract window
        eeg_window = self.eeg[start:end]
        fnirs_window = self.fnirs[start:end]
        
        # Combine channels
        features = torch.cat([eeg_window, fnirs_window], dim=1)
        
        # Remove DC offset per channel
        features = features - features.mean(dim=0, keepdim=True)
        
        # Data augmentation
        if self.augment:
            # Random scaling
            if torch.rand(1).item() < 0.3:
                scale = 0.9 + 0.2 * torch.rand(1).item()
                features = features * scale
            
            # Add small noise
            if torch.rand(1).item() < 0.3:
                noise = 0.01 * torch.randn_like(features)
                features = features + noise
        
        label = self.window_labels[idx]
        
        return features, torch.tensor([label], dtype=torch.float32)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(data_dir: str, verbose: bool = True):
    """
    Load all NPZ files and create unified dataset.
    """
    files = [f for f in os.listdir(data_dir) 
             if f.endswith('.npz') and not f.endswith('.backup')]
    
    if not files:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    if verbose:
        print(f"\n📂 Found {len(files)} data files in {data_dir}")
    
    all_eeg = []
    all_fnirs = []
    all_labels = []
    
    total_samples = 0
    total_confused = 0
    
    for fname in tqdm(files, desc="Loading files"):
        fpath = os.path.join(data_dir, fname)
        data = np.load(fpath, allow_pickle=True)
        
        eeg = data['eeg']
        fnirs = data['fnirs']
        
        # Create confusion labels based on word position overlap
        word_char_starts = data['word_char_starts']
        word_char_ends = data['word_char_ends']
        word_text_indices = data['word_text_indices']
        
        event_types = data['event_types']
        event_char_starts = data['event_char_starts']
        event_char_ends = data['event_char_ends']
        event_text_indices = data['event_text_indices']
        
        # Label samples based on confusion event overlap
        labels = np.zeros(len(eeg), dtype=np.float32)
        
        for evt_type, evt_text_idx, evt_start, evt_end in zip(
            event_types, event_text_indices, event_char_starts, event_char_ends
        ):
            if 'confusion' in str(evt_type):
                same_text = word_text_indices == evt_text_idx
                overlaps = same_text & (word_char_starts >= evt_start) & (word_char_starts < evt_end)
                labels[overlaps] = 1.0
        
        # Filter out samples with no word tracking
        valid_mask = word_char_starts >= 0
        eeg_valid = eeg[valid_mask]
        fnirs_valid = fnirs[valid_mask]
        labels_valid = labels[valid_mask]
        
        # Handle NaN values
        fnirs_valid = np.nan_to_num(fnirs_valid, nan=0.0)
        eeg_valid = np.nan_to_num(eeg_valid, nan=0.0)
        
        all_eeg.append(eeg_valid)
        all_fnirs.append(fnirs_valid)
        all_labels.append(labels_valid)
        
        n_confused = np.sum(labels_valid)
        total_samples += len(eeg_valid)
        total_confused += n_confused
        
        if verbose:
            print(f"  {fname}: {len(eeg_valid):,} samples, {int(n_confused):,} confused ({100*n_confused/len(eeg_valid):.1f}%)")
    
    # Concatenate all data
    eeg_combined = np.vstack(all_eeg)
    fnirs_combined = np.vstack(all_fnirs)
    labels_combined = np.concatenate(all_labels)
    
    if verbose:
        print(f"\n📊 Total: {total_samples:,} samples, {int(total_confused):,} confused ({100*total_confused/total_samples:.1f}%)")
    
    return eeg_combined, fnirs_combined, labels_combined


def normalize_data(eeg: np.ndarray, fnirs: np.ndarray):
    """
    Normalize data to zero mean and unit variance per channel.
    """
    eeg_mean = np.mean(eeg, axis=0)
    eeg_std = np.std(eeg, axis=0) + 1e-8
    fnirs_mean = np.mean(fnirs, axis=0)
    fnirs_std = np.std(fnirs, axis=0) + 1e-8
    
    eeg_norm = (eeg - eeg_mean) / eeg_std
    fnirs_norm = (fnirs - fnirs_mean) / fnirs_std
    
    stats = {
        'eeg_mean': eeg_mean.tolist(),
        'eeg_std': eeg_std.tolist(),
        'fnirs_mean': fnirs_mean.tolist(),
        'fnirs_std': fnirs_std.tolist()
    }
    
    return eeg_norm, fnirs_norm, stats


def create_weighted_sampler(labels: np.ndarray, window_size: int, step_size: int):
    """
    Create weighted sampler to handle class imbalance.
    """
    n_windows = (len(labels) - window_size) // step_size + 1
    window_labels = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        window_labels[i] = float(np.mean(labels[start:end]) > 0.5)
    
    n_confused = np.sum(window_labels)
    n_normal = len(window_labels) - n_confused
    
    weight_confused = len(window_labels) / (2 * n_confused + 1e-8)
    weight_normal = len(window_labels) / (2 * n_normal + 1e-8)
    
    sample_weights = np.where(window_labels == 1, weight_confused, weight_normal)
    
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for features, labels in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp=True):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    for features, labels in tqdm(dataloader, desc="Validating", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * features.size(0)
        
        probs = torch.sigmoid(outputs)
        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(float)
    
    accuracy = np.mean(all_preds == all_labels)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    
    # Calculate F1 score
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return total_loss / len(all_labels), accuracy, auc, f1, all_labels, all_probs


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(config):
    """
    Main training function.
    """
    print("\n" + "="*70)
    print("🧠 NEUROADAPTIVE READER - CONFUSION DETECTION RNN TRAINING")
    print("="*70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Calculate window/step sizes
    sample_rate = config['sample_rate']
    window_size = int(config['window_size_sec'] * sample_rate)
    step_size = int(config['step_size_sec'] * sample_rate)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {config['model_type'].upper()}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Layers: {config['num_layers']}")
    print(f"   Window: {config['window_size_sec']}s ({window_size} samples)")
    print(f"   Step: {config['step_size_sec']}s ({step_size} samples)")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Mixed precision: {config['use_amp']}")
    
    # Load data
    print("\n" + "-"*50)
    print("📥 Loading data...")
    eeg, fnirs, labels = load_all_data(config['data_dir'])
    
    # Normalize
    print("\n🔄 Normalizing data...")
    eeg_norm, fnirs_norm, norm_stats = normalize_data(eeg, fnirs)
    
    # Split into train/val (temporal split to avoid leakage)
    n_samples = len(eeg_norm)
    split_idx = int(n_samples * 0.8)
    
    eeg_train, eeg_val = eeg_norm[:split_idx], eeg_norm[split_idx:]
    fnirs_train, fnirs_val = fnirs_norm[:split_idx], fnirs_norm[split_idx:]
    labels_train, labels_val = labels[:split_idx], labels[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(eeg_train):,} samples ({100*np.mean(labels_train):.1f}% confused)")
    print(f"   Val:   {len(eeg_val):,} samples ({100*np.mean(labels_val):.1f}% confused)")
    
    # Create datasets
    train_dataset = ConfusionDataset(
        eeg_train, fnirs_train, labels_train,
        window_size=window_size, step_size=step_size, augment=True
    )
    val_dataset = ConfusionDataset(
        eeg_val, fnirs_val, labels_val,
        window_size=window_size, step_size=step_size, augment=False
    )
    
    print(f"\n📦 Dataset windows:")
    print(f"   Train: {len(train_dataset):,}")
    print(f"   Val:   {len(val_dataset):,}")
    
    # Create data loaders
    train_sampler = create_weighted_sampler(labels_train, window_size, step_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    # Create model
    print("\n" + "-"*50)
    print("🏗️  Creating model...")
    
    model_config = {
        'type': config['model_type'],
        'input_size': config['input_size'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'bidirectional': config['bidirectional'],
        'use_attention': config['use_attention']
    }
    
    if config['model_type'].lower() == 'gru':
        model = ConfusionDetectorGRU(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )
    else:
        model = ConfusionDetectorRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            use_attention=config['use_attention']
        )
    
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Loss and optimizer
    n_pos = np.sum(train_dataset.window_labels.numpy())
    n_neg = len(train_dataset.window_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)
    print(f"   Pos weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['use_amp'] else None
    
    # Training loop
    print("\n" + "-"*50)
    print("🚀 Training...")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []
    }
    
    best_auc = 0
    best_f1 = 0
    best_epoch = 0
    epochs_no_improve = 0
    
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config['use_amp']
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_f1, val_labels, val_probs = validate(
            model, val_loader, criterion, device, config['use_amp']
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, AUC: {val_auc:.3f}, F1: {val_f1:.3f}")
        print(f"   LR: {current_lr:.2e}")
        
        # Save best model (by AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            # Convert history to Python native types for compatibility
            history_native = {
                k: [float(v) for v in vals] 
                for k, vals in history.items()
            }
            
            checkpoint = {
                'epoch': int(epoch + 1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'norm_stats': norm_stats,  # Already lists from normalize_data()
                'val_auc': float(val_auc),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
                'window_size': int(window_size),
                'step_size': int(step_size),
                'sample_rate': int(sample_rate),
                'history': history_native
            }
            
            model_path = os.path.join(config['save_dir'], config['model_name'])
            torch.save(checkpoint, model_path)
            print(f"   ✅ Saved best model (AUC={val_auc:.4f}, F1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= config['patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION")
    print("="*70)
    
    print(f"\n🏆 Best epoch: {best_epoch}")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Best F1:  {best_f1:.4f}")
    
    # Load best model for final eval
    checkpoint = torch.load(
        os.path.join(config['save_dir'], config['model_name']),
        map_location=device,
        weights_only=False  # Our checkpoint contains norm_stats dict
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, val_acc, val_auc, val_f1, val_labels, val_probs = validate(
        model, val_loader, criterion, device, config['use_amp']
    )
    
    val_preds = (np.array(val_probs) >= 0.5).astype(int)
    
    print("\n📋 Classification Report:")
    print(classification_report(val_labels, val_preds, 
                               target_names=['Not Confused', 'Confused']))
    
    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    
    axes[0,0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0,0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0,1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(history['val_auc'], linewidth=2, color='green')
    axes[1,0].axhline(y=best_auc, color='r', linestyle='--', label=f'Best: {best_auc:.3f}')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('AUC-ROC')
    axes[1,0].set_title('Validation AUC')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(history['val_f1'], linewidth=2, color='purple')
    axes[1,1].axhline(y=best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.3f}')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('F1 Score')
    axes[1,1].set_title('Validation F1')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config['save_dir'], 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Saved training plot to {plot_path}")
    
    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(val_labels, val_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {val_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(config['save_dir'], 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Model saved to: {os.path.join(config['save_dir'], config['model_name'])}")
    print("\n💡 Download the model file and use it with liveeeg.py for live predictions!")
    
    return model, history


# =============================================================================
# RUN TRAINING
# =============================================================================

if __name__ == "__main__":
    # Check if data directory exists
    if not os.path.exists(CONFIG['data_dir']):
        print(f"❌ Data directory not found: {CONFIG['data_dir']}")
        print("\n📤 Please upload your NewRecordings folder to /content/NewRecordings/")
        print("   You can use the Colab file browser or run:")
        print("   !mkdir -p /content/NewRecordings")
        print("   Then upload your .npz files")
    else:
        # Run training
        model, history = train_model(CONFIG)

