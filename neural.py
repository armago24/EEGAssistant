#!/usr/bin/env python3
"""
Neural Network-based Word Confusion Detector for EEG/fNIRS Data

This module implements a deep learning approach for detecting confusion during reading
using multimodal brain signals (EEG, fNIRS) and motion data.

Key Features:
- Hybrid architecture combining CNN for raw signals and dense layers for features
- Multi-scale temporal processing for different signal modalities
- Attention mechanisms for channel/feature importance
- Robust handling of class imbalance and small datasets
- Comprehensive evaluation metrics and visualizations
- Multi-file training support
- Command-line interface
"""

"""
KEEP IN MIND THAT THIS SYSTEM WILL WORK WITHOUT LOADING THE WORDS THEMSELVES. THEREFORE, I WOULDN'T KNOW
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from collections import Counter, defaultdict
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Signal processing
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

# Machine learning utilities
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Progress tracking
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG/fNIRS confusion detection"""
    
    def __init__(self, raw_signals, features, labels, augment=False):
        """
        Args:
            raw_signals: Dict with 'eeg', 'fnirs', 'motion' arrays
            features: Engineered features array
            labels: Class labels (0=baseline, 1=word_confusion, 2=sentence_confusion)
            augment: Whether to apply data augmentation
        """
        self.raw_signals = raw_signals
        self.features = features
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get raw signals
        eeg = self.raw_signals['eeg'][idx].copy()
        fnirs = self.raw_signals['fnirs'][idx].copy()
        motion = self.raw_signals['motion'][idx].copy()
        
        # Get engineered features
        features = self.features[idx].copy()
        
        # Get label
        label = self.labels[idx]
        
        # Apply augmentation if training
        if self.augment and np.random.rand() > 0.5:
            # Time shift augmentation
            shift = np.random.randint(-10, 10)
            eeg = np.roll(eeg, shift, axis=1)
            fnirs = np.roll(fnirs, shift, axis=1)
            
            # Noise augmentation
            if np.random.rand() > 0.5:
                noise_scale = 0.05
                eeg += np.random.randn(*eeg.shape) * noise_scale * np.std(eeg)
                fnirs += np.random.randn(*fnirs.shape) * noise_scale * np.std(fnirs)
            
            # Channel dropout
            if np.random.rand() > 0.7:
                dropout_ch = np.random.randint(0, eeg.shape[0])
                eeg[dropout_ch] *= 0.1
        
        return {
            'eeg': torch.FloatTensor(eeg),
            'fnirs': torch.FloatTensor(fnirs),
            'motion': torch.FloatTensor(motion),
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label])[0]
        }


class ChannelAttention(nn.Module):
    """Channel attention mechanism for learning channel importance"""
    
    def __init__(self, n_channels):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, n_channels // 2)
        self.fc2 = nn.Linear(n_channels // 2, n_channels)
        
    def forward(self, x):
        # Global average pooling across time
        avg_pool = torch.mean(x, dim=2)
        
        # Attention weights
        attn = F.relu(self.fc1(avg_pool))
        attn = torch.sigmoid(self.fc2(attn))
        
        # Apply attention
        return x * attn.unsqueeze(2)


class TemporalConvBlock(nn.Module):
    """1D Convolutional block for temporal feature extraction"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, dilation=dilation, 
                            padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleEEGEncoder(nn.Module):
    """Multi-scale CNN encoder for EEG signals"""
    
    def __init__(self, n_channels=4, n_timepoints=512):
        super().__init__()
        
        # Multi-scale convolutional branches
        self.conv_3 = TemporalConvBlock(n_channels, 32, kernel_size=3)
        self.conv_5 = TemporalConvBlock(n_channels, 32, kernel_size=5)
        self.conv_7 = TemporalConvBlock(n_channels, 32, kernel_size=7)
        
        # Deeper layers
        self.conv2 = TemporalConvBlock(96, 64, kernel_size=3, stride=2)
        self.conv3 = TemporalConvBlock(64, 128, kernel_size=3, stride=2)
        
        # Channel attention
        self.channel_attn = ChannelAttention(n_channels)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # Apply channel attention
        x = self.channel_attn(x)
        
        # Multi-scale processing
        x1 = self.conv_3(x)
        x2 = self.conv_5(x)
        x3 = self.conv_7(x)
        
        # Concatenate scales
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Deeper processing
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        return x


class fNIRSEncoder(nn.Module):
    """Encoder for fNIRS signals (slower dynamics)"""
    
    def __init__(self, n_channels=8, n_timepoints=512):
        super().__init__()
        
        # Larger kernels for slower dynamics
        self.conv1 = TemporalConvBlock(n_channels, 16, kernel_size=15)
        self.conv2 = TemporalConvBlock(16, 32, kernel_size=11, stride=2)
        self.conv3 = TemporalConvBlock(32, 64, kernel_size=7, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class ConfusionDetectorNN(nn.Module):
    """Main neural network for confusion detection"""
    
    def __init__(self, n_eeg_ch=4, n_fnirs_ch=8, n_motion_ch=6, 
                 n_features=100, n_classes=3, n_timepoints=512):
        super().__init__()
        
        # Encoders for each modality
        self.eeg_encoder = MultiScaleEEGEncoder(n_eeg_ch, n_timepoints)
        self.fnirs_encoder = fNIRSEncoder(n_fnirs_ch, n_timepoints)
        
        # Motion encoder (simple, as it's auxiliary)
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(n_motion_ch, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion dimensions
        fusion_dim = 128 + 64 + 16 + 64  # EEG + fNIRS + motion + features
        
        # Attention-based fusion
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, eeg, fnirs, motion, features):
        # Encode each modality
        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)
        motion_feat = self.motion_encoder(motion)
        feat_encoded = self.feature_encoder(features)
        
        # Concatenate all features
        fused = torch.cat([eeg_feat, fnirs_feat, motion_feat, feat_encoded], dim=1)
        
        # Apply attention-based fusion
        attn_weights = self.fusion_attention(fused)
        fused = fused * attn_weights
        
        # Classification
        output = self.classifier(fused)
        
        return output, attn_weights


class WordConfusionDetectorNN:
    """Complete neural network pipeline for word-level confusion detection"""
    
    def __init__(self, model_name="confusion_detector_nn"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Signal processing parameters
        self.sample_rate = 256  # Hz for EEG
        self.window_size = 2.0  # seconds
        self.window_samples = int(self.window_size * self.sample_rate)
        
        # Training history
        self.history = defaultdict(list)
        
        # Store data from multiple files
        self.all_data = {
            'timestamps': [],
            'eeg_data': [],
            'fnirs_data': [],
            'motion_data': [],
            'event_timestamps': [],
            'event_types': [],
            'event_words': [],
            'tracked_words': []
        }
        
    def load_multiple_files(self, filepaths):
        """Load and combine data from multiple NPZ files"""
        print(f"\nLoading data from {len(filepaths)} files...")
        
        cumulative_time_offset = 0
        
        for filepath in filepaths:
            print(f"\nProcessing {filepath}...")
            data = np.load(filepath, allow_pickle=True)
            
            # Get timestamps for this file
            timestamps = data['timestamps']
            
            # Add time offset to avoid overlapping timestamps
            if len(self.all_data['timestamps']) > 0:
                cumulative_time_offset = self.all_data['timestamps'][-1][-1] + 10.0
            
            # Append data with time offset
            self.all_data['timestamps'].append(timestamps + cumulative_time_offset)
            self.all_data['eeg_data'].append(data['eeg'])
            self.all_data['fnirs_data'].append(data['fnirs'])
            self.all_data['motion_data'].append(data['motion'])
            
            # Append events with time offset
            self.all_data['event_timestamps'].append(data['event_timestamps'] + cumulative_time_offset)
            self.all_data['event_types'].append(data['event_types'])
            self.all_data['event_words'].append(data['event_words'])
            
            # Handle words - they might be in different formats
            if 'words' in data:
                words = data['words']
                if len(words) > 0:
                    # Check if words are already dicts or need to be converted
                    processed_words = []
                    for word_entry in words:
                        if isinstance(word_entry, dict):
                            # Already a dict, just offset the timestamp
                            word_copy = word_entry.copy()
                            if 'timestamp' in word_copy:
                                word_copy['timestamp'] += cumulative_time_offset
                            processed_words.append(word_copy)
                        else:
                            # Try to extract word info from other formats
                            # This handles cases where words might be stored differently
                            pass
                    self.all_data['tracked_words'].extend(processed_words)
            
            print(f"  Loaded {len(timestamps)} samples from {filepath}")
        
        # Concatenate all data
        self.timestamps = np.concatenate(self.all_data['timestamps'])
        self.eeg_data = np.vstack(self.all_data['eeg_data'])
        self.fnirs_data = np.vstack(self.all_data['fnirs_data'])
        self.motion_data = np.vstack(self.all_data['motion_data'])
        
        self.event_timestamps = np.concatenate(self.all_data['event_timestamps'])
        self.event_types = np.concatenate(self.all_data['event_types'])
        self.event_words = np.concatenate(self.all_data['event_words'])
        
        self.tracked_words = self.all_data['tracked_words']
        
        print(f"\nTotal loaded:")
        print(f"  Samples: {len(self.timestamps)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  fNIRS shape: {self.fnirs_data.shape}")
        print(f"  Events: {len(self.event_timestamps)}")
        print(f"  Tracked words: {len(self.tracked_words)}")
        
    def create_dataset(self):
        """Create training dataset with word-level labels"""
        print("\nCreating dataset...")
        
        # Parse confusion events
        confused_words = set()
        confusion_map = defaultdict(int)
        
        for i, (ts, event_type, words) in enumerate(zip(
            self.event_timestamps, self.event_types, self.event_words)):
            
            words_list = words.split() if isinstance(words, str) else [words]
            for word in words_list:
                confused_words.add((word.lower(), ts))
                confusion_map[word.lower()] += 1
        
        # Create samples
        samples = []
        raw_signals = {'eeg': [], 'fnirs': [], 'motion': []}
        features = []
        labels = []
        
        # If we have tracked words, use them
        if len(self.tracked_words) > 0:
            print(f"Processing {len(self.tracked_words)} tracked words...")
            
            for word_info in tqdm(self.tracked_words):
                if isinstance(word_info, dict) and 'timestamp' in word_info:
                    word_ts = word_info['timestamp']
                    word_text = word_info.get('text', '').lower()
                else:
                    continue
                
                # Find closest sample index
                idx = np.argmin(np.abs(self.timestamps - word_ts))
                
                # Extract window
                half_window = self.window_samples // 2
                start_idx = max(0, idx - half_window)
                end_idx = min(len(self.timestamps), idx + half_window)
                
                if end_idx - start_idx < self.window_samples // 2:
                    continue
                
                # Extract signals
                eeg_win = self.eeg_filtered[start_idx:end_idx].T
                fnirs_win = self.fnirs_filtered[start_idx:end_idx].T
                motion_win = self.motion_data[start_idx:end_idx].T
                
                # Pad or truncate to fixed size
                target_len = self.window_samples
                eeg_win = self._pad_or_truncate(eeg_win, target_len)
                fnirs_win = self._pad_or_truncate(fnirs_win, target_len)
                motion_win = self._pad_or_truncate(motion_win, target_len)
                
                # Determine label
                label = 0  # baseline
                
                # Check if word was marked as confusing
                for conf_word, conf_ts in confused_words:
                    if abs(word_ts - conf_ts) < 3.0:  # Within 3 seconds
                        if word_text == conf_word:
                            # Check event type
                            event_idx = np.argmin(np.abs(self.event_timestamps - conf_ts))
                            if 'sentence' in self.event_types[event_idx]:
                                label = 2  # sentence confusion
                            else:
                                label = 1  # word confusion
                            break
                
                # Skip if too close to confusion but not confused (ambiguous)
                if label == 0:
                    min_dist_to_confusion = float('inf')
                    for _, conf_ts in confused_words:
                        min_dist_to_confusion = min(min_dist_to_confusion, abs(word_ts - conf_ts))
                    
                    if min_dist_to_confusion < 3.0:
                        continue  # Skip ambiguous samples
                
                # Extract features
                word_features = {
                    'text': word_text,
                    'previously_confused': int(word_text in confusion_map),
                    'confusion_count': confusion_map.get(word_text, 0)
                }
                
                feat_vec = self.extract_features(eeg_win, fnirs_win, word_features)
                
                # Store
                raw_signals['eeg'].append(eeg_win)
                raw_signals['fnirs'].append(fnirs_win)
                raw_signals['motion'].append(motion_win)
                features.append(feat_vec)
                labels.append(label)
        
        else:
            # Fallback: create event-based samples if no word timeline
            print("No word timeline found. Creating event-based samples...")
            
            # Create confusion event samples
            for i, (event_ts, event_type, event_text) in enumerate(zip(
                self.event_timestamps, self.event_types, self.event_words)):
                
                # Find closest sample index
                idx = np.argmin(np.abs(self.timestamps - event_ts))
                
                # Extract window around event
                half_window = self.window_samples // 2
                start_idx = max(0, idx - half_window)
                end_idx = min(len(self.timestamps), idx + half_window)
                
                if end_idx - start_idx < self.window_samples // 2:
                    continue
                
                # Extract signals
                eeg_win = self.eeg_filtered[start_idx:end_idx].T
                fnirs_win = self.fnirs_filtered[start_idx:end_idx].T
                motion_win = self.motion_data[start_idx:end_idx].T
                
                # Pad or truncate to fixed size
                eeg_win = self._pad_or_truncate(eeg_win, self.window_samples)
                fnirs_win = self._pad_or_truncate(fnirs_win, self.window_samples)
                motion_win = self._pad_or_truncate(motion_win, self.window_samples)
                
                # Determine label based on event type
                if 'sentence' in event_type:
                    label = 2  # sentence confusion
                else:
                    label = 1  # word confusion
                
                # Extract features
                feat_vec = self.extract_features(eeg_win, fnirs_win, None)
                
                # Store
                raw_signals['eeg'].append(eeg_win)
                raw_signals['fnirs'].append(fnirs_win)
                raw_signals['motion'].append(motion_win)
                features.append(feat_vec)
                labels.append(label)
            
            # Create baseline samples (random windows far from events)
            n_baseline_needed = len(labels) * 2  # 2:1 ratio
            baseline_added = 0
            
            while baseline_added < n_baseline_needed:
                # Random timestamp
                random_idx = np.random.randint(self.window_samples, len(self.timestamps) - self.window_samples)
                random_ts = self.timestamps[random_idx]
                
                # Check distance from all events
                min_dist_to_event = float('inf')
                for event_ts in self.event_timestamps:
                    min_dist_to_event = min(min_dist_to_event, abs(random_ts - event_ts))
                
                # Only use if far enough from events
                if min_dist_to_event > 5.0:  # At least 5 seconds away
                    # Extract window
                    half_window = self.window_samples // 2
                    start_idx = random_idx - half_window
                    end_idx = random_idx + half_window
                    
                    # Extract signals
                    eeg_win = self.eeg_filtered[start_idx:end_idx].T
                    fnirs_win = self.fnirs_filtered[start_idx:end_idx].T
                    motion_win = self.motion_data[start_idx:end_idx].T
                    
                    # Pad or truncate to fixed size
                    eeg_win = self._pad_or_truncate(eeg_win, self.window_samples)
                    fnirs_win = self._pad_or_truncate(fnirs_win, self.window_samples)
                    motion_win = self._pad_or_truncate(motion_win, self.window_samples)
                    
                    # Extract features
                    feat_vec = self.extract_features(eeg_win, fnirs_win, None)
                    
                    # Store
                    raw_signals['eeg'].append(eeg_win)
                    raw_signals['fnirs'].append(fnirs_win)
                    raw_signals['motion'].append(motion_win)
                    features.append(feat_vec)
                    labels.append(0)  # baseline
                    
                    baseline_added += 1
        
        # Convert to arrays
        for key in raw_signals:
            raw_signals[key] = np.array(raw_signals[key])
        features = np.array(features)
        labels = np.array(labels)
        
        # Print class distribution
        print(f"\nClass distribution:")
        for label, count in Counter(labels).items():
            label_name = ['baseline', 'word_confusion', 'sentence_confusion'][label]
            print(f"  {label_name}: {count} ({count/len(labels)*100:.1f}%)")
        
        if len(labels) == 0:
            raise ValueError("No valid samples could be created from the data. "
                           "Check that the NPZ files contain proper event data.")
        
        return raw_signals, features, labels
    
    def preprocess_signals(self):
        """Apply signal preprocessing"""
        print("\nPreprocessing signals...")
        
        # EEG preprocessing
        # Bandpass filter 0.5-50 Hz
        sos = signal.butter(4, [0.5, 50], btype='band', fs=self.sample_rate, output='sos')
        self.eeg_filtered = signal.sosfiltfilt(sos, self.eeg_data, axis=0)
        
        # Notch filters for line noise
        for freq in [60, 120]:
            sos_notch = signal.butter(4, [freq-2, freq+2], btype='bandstop', 
                                    fs=self.sample_rate, output='sos')
            self.eeg_filtered = signal.sosfiltfilt(sos_notch, self.eeg_filtered, axis=0)
        
        # fNIRS preprocessing (only normalized channels)
        # Low-pass filter at 0.5 Hz for hemodynamic response
        sos_fnirs = signal.butter(4, 0.5, btype='low', fs=10, output='sos')  # fNIRS is slower
        self.fnirs_filtered = self.fnirs_data[:, :4].copy()  # Use normalized channels
        
        # Detrend
        self.fnirs_filtered = signal.detrend(self.fnirs_filtered, axis=0)
        
        print("Signal preprocessing complete")
        
    def extract_features(self, eeg_window, fnirs_window, word_info=None):
        """Extract hand-crafted features from signal windows"""
        features = []
        
        # EEG features (per channel)
        for ch in range(eeg_window.shape[0]):
            ch_data = eeg_window[ch]
            
            # Time domain
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(np.abs(ch_data)),
                skew(ch_data),
                kurtosis(ch_data)
            ])
            
            # Frequency domain
            freqs, psd = signal.welch(ch_data, fs=self.sample_rate, nperseg=min(256, len(ch_data)))
            
            # Band powers
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features.append(band_power)
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            features.append(peak_freq)
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-15))
            features.append(spectral_entropy)
        
        # fNIRS features (per channel)
        for ch in range(fnirs_window.shape[0]):
            ch_data = fnirs_window[ch]
            
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data) - np.min(ch_data),  # Range
                np.polyfit(np.arange(len(ch_data)), ch_data, 1)[0],  # Slope
                np.argmax(ch_data) / len(ch_data)  # Time to peak (normalized)
            ])
        
        # Connectivity features
        # Frontal asymmetry (AF7 - AF8 alpha power)
        af7_alpha = self._get_band_power(eeg_window[1], 'alpha')
        af8_alpha = self._get_band_power(eeg_window[2], 'alpha')
        features.append(af7_alpha - af8_alpha)
        
        # Inter-channel coherence
        channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for ch1, ch2 in channel_pairs:
            coh = self._compute_coherence(eeg_window[ch1], eeg_window[ch2])
            features.append(coh)
        
        # Word features (if provided)
        if word_info:
            features.extend([
                len(word_info.get('text', '')),
                int(any(c.isdigit() for c in word_info.get('text', ''))),
                word_info.get('previously_confused', 0),
                word_info.get('confusion_count', 0)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _get_band_power(self, signal_data, band_name):
        """Helper to compute band power"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        freqs, psd = signal.welch(signal_data, fs=self.sample_rate, nperseg=min(256, len(signal_data)))
        low, high = bands[band_name]
        band_mask = (freqs >= low) & (freqs < high)
        return np.trapz(psd[band_mask], freqs[band_mask])
    
    def _compute_coherence(self, signal1, signal2):
        """Compute coherence between two signals"""
        f, Cxy = signal.coherence(signal1, signal2, fs=self.sample_rate, 
                                nperseg=min(64, len(signal1)))
        # Return mean coherence in alpha band
        alpha_mask = (f >= 8) & (f <= 13)
        return np.mean(Cxy[alpha_mask])
    
    def _pad_or_truncate(self, signal, target_len):
        """Pad or truncate signal to target length"""
        if signal.shape[1] > target_len:
            return signal[:, :target_len]
        elif signal.shape[1] < target_len:
            pad_len = target_len - signal.shape[1]
            return np.pad(signal, ((0, 0), (0, pad_len)), mode='edge')
        return signal
    
    def train_model(self, raw_signals, features, labels, n_epochs=100, batch_size=32):
        """Train the neural network"""
        print("\n" + "="*50)
        print("TRAINING NEURAL NETWORK")
        print("="*50)
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, 
                                            stratify=labels, random_state=SEED)
        
        # Scale features
        self.scaler.fit(features[train_idx])
        features_scaled = self.scaler.transform(features)
        
        # Create datasets
        train_dataset = EEGDataset(
            {k: v[train_idx] for k, v in raw_signals.items()},
            features_scaled[train_idx],
            labels[train_idx],
            augment=True
        )
        
        val_dataset = EEGDataset(
            {k: v[val_idx] for k, v in raw_signals.items()},
            features_scaled[val_idx],
            labels[val_idx],
            augment=False
        )
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(labels), 
                                           y=labels[train_idx])
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Create weighted sampler for balanced training
        train_labels = labels[train_idx]
        sample_weights = np.zeros(len(train_labels))
        for i, label in enumerate(train_labels):
            sample_weights[i] = class_weights[label].item()
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=0)
        
        # Initialize model
        self.model = ConfusionDetectorNN(
            n_eeg_ch=raw_signals['eeg'].shape[1],
            n_fnirs_ch=raw_signals['fnirs'].shape[1],
            n_motion_ch=raw_signals['motion'].shape[1],
            n_features=features.shape[1],
            n_classes=3,
            n_timepoints=raw_signals['eeg'].shape[2]
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]"):
                # Move to device
                eeg = batch['eeg'].to(self.device)
                fnirs = batch['fnirs'].to(self.device)
                motion = batch['motion'].to(self.device)
                feat = batch['features'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = self.model(eeg, fnirs, motion, feat)
                loss = criterion(outputs, labels_batch)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(labels_batch.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(self.device)
                    fnirs = batch['fnirs'].to(self.device)
                    motion = batch['motion'].to(self.device)
                    feat = batch['features'].to(self.device)
                    labels_batch = batch['label'].to(self.device)
                    
                    outputs, _ = self.model(eeg, fnirs, motion, feat)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(labels_batch.cpu().numpy())
                    val_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
            val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"\nEpoch {epoch+1}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pth'))
        
        # Final evaluation
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
        # Get final predictions
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        full_dataset = EEGDataset(raw_signals, features_scaled, labels, augment=False)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in full_loader:
                eeg = batch['eeg'].to(self.device)
                fnirs = batch['fnirs'].to(self.device)
                motion = batch['motion'].to(self.device)
                feat = batch['features'].to(self.device)
                
                outputs, _ = self.model(eeg, fnirs, motion, feat)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Print classification report
        class_names = ['Baseline', 'Word Confusion', 'Sentence Confusion']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                  target_names=class_names, digits=3))
        
        # Calculate additional metrics
        cm = confusion_matrix(all_labels, all_preds)
        
        # Per-class accuracy
        print("\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            class_mask = all_labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(all_preds[class_mask] == i)
                class_prec = cm[i, i] / (np.sum(cm[:, i]) + 1e-10)
                class_recall = cm[i, i] / (np.sum(cm[i, :]) + 1e-10)
                print(f"  {class_name}: Acc={class_acc:.3f}, "
                      f"Prec={class_prec:.3f}, Recall={class_recall:.3f}")
        
        # Confusion detection performance (combining both confusion classes)
        confusion_labels = (all_labels > 0).astype(int)
        confusion_preds = (all_preds > 0).astype(int)
        confusion_probs = all_probs[:, 1] + all_probs[:, 2]
        
        confusion_acc = np.mean(confusion_labels == confusion_preds)
        confusion_cm = confusion_matrix(confusion_labels, confusion_preds)
        
        print(f"\nBinary Confusion Detection Performance:")
        print(f"  Accuracy: {confusion_acc:.3f}")
        print(f"  Sensitivity: {confusion_cm[1, 1] / np.sum(confusion_cm[1, :]):.3f}")
        print(f"  Specificity: {confusion_cm[0, 0] / np.sum(confusion_cm[0, :]):.3f}")
        
        try:
            auc_score = roc_auc_score(confusion_labels, confusion_probs)
            print(f"  AUC: {auc_score:.3f}")
        except:
            print("  AUC: Could not calculate (likely single class in data)")
        
        return all_preds, all_labels, all_probs
    
    def visualize_results(self, predictions, labels, probabilities):
        """Create comprehensive visualization of results"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Training history
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=['Baseline', 'Word', 'Sentence'],
                   yticklabels=['Baseline', 'Word', 'Sentence'],
                   ax=ax3, cbar_kws={'label': 'Proportion'})
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # 3. ROC Curves
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Binary confusion detection ROC
        confusion_labels = (labels > 0).astype(int)
        confusion_scores = probabilities[:, 1] + probabilities[:, 2]
        
        try:
            fpr, tpr, _ = roc_curve(confusion_labels, confusion_scores)
            auc_score = auc(fpr, tpr)
            ax4.plot(fpr, tpr, label=f'Confusion Detection (AUC={auc_score:.3f})', 
                    linewidth=2)
        except:
            pass
        
        # Multi-class ROC
        for i, class_name in enumerate(['Baseline', 'Word Conf', 'Sentence Conf']):
            try:
                class_labels = (labels == i).astype(int)
                class_scores = probabilities[:, i]
                fpr, tpr, _ = roc_curve(class_labels, class_scores)
                auc_score = auc(fpr, tpr)
                ax4.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.2f})', 
                        linewidth=2, alpha=0.7)
            except:
                pass
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        # 4. Class Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        class_counts = np.bincount(labels)
        bars = ax5.bar(['Baseline', 'Word', 'Sentence'], class_counts, 
                       color=['#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax5.set_ylabel('Count')
        ax5.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 5. Prediction Confidence Distribution
        ax6 = fig.add_subplot(gs[2, :])
        
        # Get confidence scores for predicted classes
        confidence_scores = []
        for i, pred in enumerate(predictions):
            confidence_scores.append(probabilities[i, pred])
        
        # Plot by true class
        for class_idx, class_name in enumerate(['Baseline', 'Word Confusion', 'Sentence Confusion']):
            mask = labels == class_idx
            if np.sum(mask) > 0:
                class_confidences = [confidence_scores[i] for i in range(len(labels)) if mask[i]]
                ax6.hist(class_confidences, bins=20, alpha=0.6, label=class_name, 
                        density=True)
        
        ax6.set_xlabel('Prediction Confidence')
        ax6.set_ylabel('Density')
        ax6.set_title('Prediction Confidence Distribution by True Class', 
                     fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 6. Performance Metrics Summary
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Calculate metrics
        overall_acc = np.mean(predictions == labels)
        
        # Per-class metrics
        metrics_text = "PERFORMANCE SUMMARY\n" + "="*50 + "\n\n"
        metrics_text += f"Overall Accuracy: {overall_acc:.3f}\n\n"
        
        cm = confusion_matrix(labels, predictions)
        for i, class_name in enumerate(['Baseline', 'Word Confusion', 'Sentence Confusion']):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics_text += f"{class_name}:\n"
                metrics_text += f"  Precision: {precision:.3f}\n"
                metrics_text += f"  Recall: {recall:.3f}\n"
                metrics_text += f"  F1-Score: {f1:.3f}\n"
                metrics_text += f"  Support: {np.sum(class_mask)}\n\n"
        
        # Binary confusion metrics
        confusion_binary = (labels > 0).astype(int)
        confusion_pred_binary = (predictions > 0).astype(int)
        cm_binary = confusion_matrix(confusion_binary, confusion_pred_binary)
        
        metrics_text += "Binary Confusion Detection:\n"
        if np.sum(cm_binary[1,:]) > 0:
            metrics_text += f"  Sensitivity: {cm_binary[1,1]/np.sum(cm_binary[1,:]):.3f}\n"
        if np.sum(cm_binary[0,:]) > 0:
            metrics_text += f"  Specificity: {cm_binary[0,0]/np.sum(cm_binary[0,:]):.3f}\n"
        
        ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, 
                fontfamily='monospace', fontsize=12, verticalalignment='top')
        
        plt.suptitle('EEG/fNIRS Confusion Detection - Neural Network Results', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.model_name}_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, filepath=None):
        """Save the complete model package for real-time deployment"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'{self.model_name}_{timestamp}.pth'
        
        # Save complete model package
        model_package = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_eeg_ch': 4,
                'n_fnirs_ch': 8,
                'n_motion_ch': 6,
                'n_features': self.scaler.mean_.shape[0],
                'n_classes': 3,
                'n_timepoints': self.window_samples
            },
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'history': dict(self.history),
            'sample_rate': self.sample_rate,
            'window_size': self.window_size,
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'model_type': 'neural_network',
                'framework': 'pytorch',
                'version': '1.0'
            }
        }
        
        torch.save(model_package, filepath)
        print(f"\nModel saved to: {filepath}")
        
        # Also save as ONNX for deployment
        onnx_path = filepath.replace('.pth', '.onnx')
        self.export_onnx(onnx_path)
        
        # Save a simplified inference script
        self.save_inference_script(filepath)
        
        return filepath
    
    def export_onnx(self, filepath):
        """Export model to ONNX format for deployment"""
        print(f"Exporting to ONNX format: {filepath}")
        
        self.model.eval()
        
        # Create dummy inputs
        dummy_eeg = torch.randn(1, 4, self.window_samples).to(self.device)
        dummy_fnirs = torch.randn(1, 8, self.window_samples).to(self.device)
        dummy_motion = torch.randn(1, 6, self.window_samples).to(self.device)
        dummy_features = torch.randn(1, self.scaler.mean_.shape[0]).to(self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            (dummy_eeg, dummy_fnirs, dummy_motion, dummy_features),
            filepath,
            input_names=['eeg', 'fnirs', 'motion', 'features'],
            output_names=['predictions', 'attention_weights'],
            dynamic_axes={
                'eeg': {0: 'batch_size'},
                'fnirs': {0: 'batch_size'},
                'motion': {0: 'batch_size'},
                'features': {0: 'batch_size'},
                'predictions': {0: 'batch_size'},
                'attention_weights': {0: 'batch_size'}
            },
            opset_version=11
        )
        
        print(f"ONNX model exported successfully")
    
    def save_inference_script(self, model_path):
        """Save a simplified script for real-time inference"""
        inference_code = '''#!/usr/bin/env python3
"""
Real-time inference script for EEG/fNIRS confusion detection
Auto-generated from trained model
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.stats import skew, kurtosis

class RealTimeConfusionDetector:
    def __init__(self, model_path):
        # Load model package
        self.device = torch.device('cpu')  # Use CPU for real-time
        package = torch.load(model_path, map_location=self.device)
        
        # Load components
        self.model_config = package['model_config']
        self.scaler = package['scaler']
        self.sample_rate = package['sample_rate']
        self.window_size = package['window_size']
        self.window_samples = int(self.window_size * self.sample_rate)
        
        # Recreate and load model
        from newneuraltrainer import ConfusionDetectorNN
        self.model = ConfusionDetectorNN(**self.model_config).to(self.device)
        self.model.load_state_dict(package['model_state_dict'])
        self.model.eval()
        
        # Initialize buffers
        self.reset_buffers()
        
    def reset_buffers(self):
        """Reset data buffers"""
        self.eeg_buffer = []
        self.fnirs_buffer = []
        self.motion_buffer = []
        
    def predict_confusion(self, eeg_data, fnirs_data, motion_data, word_info=None):
        """
        Real-time confusion prediction
        
        Args:
            eeg_data: (4, n_samples) EEG data
            fnirs_data: (8, n_samples) fNIRS data
            motion_data: (6, n_samples) Motion data
            word_info: Optional dict with word metadata
            
        Returns:
            confusion_probability: Float between 0-1
            class_probabilities: [baseline, word_conf, sentence_conf]
        """
        # Preprocess signals
        eeg_processed = self._preprocess_eeg(eeg_data)
        fnirs_processed = self._preprocess_fnirs(fnirs_data)
        
        # Extract features
        features = self._extract_features(eeg_processed, fnirs_processed, word_info)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Prepare tensors
        eeg_tensor = torch.FloatTensor(eeg_processed).unsqueeze(0)
        fnirs_tensor = torch.FloatTensor(fnirs_processed).unsqueeze(0)
        motion_tensor = torch.FloatTensor(motion_data).unsqueeze(0)
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        with torch.no_grad():
            outputs, _ = self.model(eeg_tensor, fnirs_tensor, 
                                  motion_tensor, features_tensor)
            probs = F.softmax(outputs, dim=1).numpy()[0]
        
        # Calculate confusion probability
        confusion_prob = probs[1] + probs[2]
        
        return confusion_prob, probs
    
    def _preprocess_eeg(self, eeg_data):
        """Preprocess EEG data"""
        # Bandpass filter
        sos = signal.butter(4, [0.5, 50], btype='band', 
                          fs=self.sample_rate, output='sos')
        filtered = signal.sosfiltfilt(sos, eeg_data, axis=1)
        
        # Notch filter
        for freq in [60, 120]:
            sos_notch = signal.butter(4, [freq-2, freq+2], btype='bandstop', 
                                    fs=self.sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos_notch, filtered, axis=1)
        
        return filtered
    
    def _preprocess_fnirs(self, fnirs_data):
        """Preprocess fNIRS data"""
        # Use only normalized channels
        normalized = fnirs_data[:4]
        
        # Detrend
        detrended = signal.detrend(normalized, axis=1)
        
        return detrended
    
    def _extract_features(self, eeg_data, fnirs_data, word_info):
        """Extract features matching training pipeline"""
        # This is a simplified version - implement full feature extraction
        # matching the training script for production use
        features = []
        
        # Add EEG features
        for ch in range(eeg_data.shape[0]):
            features.extend([
                np.mean(eeg_data[ch]),
                np.std(eeg_data[ch]),
                np.max(np.abs(eeg_data[ch])),
                skew(eeg_data[ch]),
                kurtosis(eeg_data[ch])
            ])
        
        # Add fNIRS features
        for ch in range(fnirs_data.shape[0]):
            features.extend([
                np.mean(fnirs_data[ch]),
                np.std(fnirs_data[ch]),
                np.max(fnirs_data[ch]) - np.min(fnirs_data[ch])
            ])
        
        # Pad to match expected feature count
        while len(features) < self.scaler.mean_.shape[0]:
            features.append(0)
        
        return np.array(features[:self.scaler.mean_.shape[0]])

# Example usage
if __name__ == "__main__":
    detector = RealTimeConfusionDetector("''' + model_path + '''")
    
    # Simulate real-time data
    dummy_eeg = np.random.randn(4, 512)
    dummy_fnirs = np.random.randn(8, 512)
    dummy_motion = np.random.randn(6, 512)
    
    confusion_prob, class_probs = detector.predict_confusion(
        dummy_eeg, dummy_fnirs, dummy_motion
    )
    
    print(f"Confusion probability: {confusion_prob:.3f}")
    print(f"Class probabilities: {class_probs}")
'''
        
        # Save inference script
        inference_path = model_path.replace('.pth', '_inference.py')
        with open(inference_path, 'w') as f:
            f.write(inference_code)
        
        print(f"Inference script saved to: {inference_path}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Train neural network for EEG/fNIRS word confusion detection'
    )
    parser.add_argument(
        'data_files',
        nargs='+',
        help='Path(s) to NPZ data files'
    )
    parser.add_argument(
        '--model-name',
        default='confusion_detector_nn',
        help='Name for the model (default: confusion_detector_nn)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save outputs (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    data_files = []
    for filepath in args.data_files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
        if not filepath.endswith('.npz'):
            print(f"Error: Expected NPZ file, got: {filepath}")
            sys.exit(1)
        data_files.append(path)
    
    print(f"Found {len(data_files)} data file(s)")
    
    # Initialize detector
    detector = WordConfusionDetectorNN(model_name=args.model_name)
    
    # Load all data files
    detector.load_multiple_files(data_files)
    
    # Preprocess signals
    detector.preprocess_signals()
    
    # Create dataset
    try:
        raw_signals, features, labels = detector.create_dataset()
    except ValueError as e:
        print(f"\nError creating dataset: {e}")
        print("\nPlease ensure your NPZ files contain:")
        print("  - Proper event data (event_timestamps, event_types, event_words)")
        print("  - Or word timeline data (words array with timestamp info)")
        sys.exit(1)
    
    # Train model
    predictions, true_labels, probabilities = detector.train_model(
        raw_signals, features, labels, 
        n_epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Visualize results
    detector.visualize_results(predictions, true_labels, probabilities)
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = detector.save_model(
        str(output_path / f"{args.model_name}_trained.pth")
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"You can now use the model for real-time inference")
    print(f"See the generated inference script for usage examples")


if __name__ == "__main__":
    main()