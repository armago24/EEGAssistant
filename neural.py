#!/usr/bin/env python3
"""
Neural Network-based Word Confusion Detector for EEG/fNIRS Data
Fixed version addressing data leakage and adding feature importance visualization
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
import re
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
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold
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


class ReadingSegmentDetector:
    """Detect reading vs non-reading segments based on signal characteristics"""
    
    def __init__(self, sample_rate=256):
        self.sample_rate = sample_rate
        
    def detect_reading_segments(self, eeg_data, motion_data, event_timestamps):
        """Identify periods of active reading vs labeling/breaks"""
        # Calculate motion energy in sliding windows
        window_size = int(0.5 * self.sample_rate)  # 0.5 second windows
        motion_energy = []
        
        for i in range(0, len(motion_data) - window_size, window_size // 2):
            window = motion_data[i:i+window_size]
            energy = np.sum(np.std(window, axis=0))
            motion_energy.append(energy)
        
        motion_energy = np.array(motion_energy)
        motion_threshold = np.percentile(motion_energy, 75)  # High motion = likely clicking/labeling
        
        # Create reading mask
        reading_mask = np.ones(len(motion_data), dtype=bool)
        
        # Mark high motion periods as non-reading
        for i, energy in enumerate(motion_energy):
            if energy > motion_threshold:
                start = i * (window_size // 2)
                end = min(start + window_size, len(reading_mask))
                reading_mask[start:end] = False
        
        # Mark periods around events as non-reading (labeling time)
        for event_ts in event_timestamps:
            event_idx = np.argmin(np.abs(self.timestamps - event_ts))
            # Mark 5 seconds before and after as labeling period
            start = max(0, event_idx - int(5 * self.sample_rate))
            end = min(len(reading_mask), event_idx + int(5 * self.sample_rate))
            reading_mask[start:end] = False
        
        return reading_mask


class WordComplexityAnalyzer:
    """Analyze word complexity features"""
    
    def __init__(self):
        # Common English digraphs and trigraphs that are difficult
        self.difficult_patterns = [
            'ph', 'gh', 'ght', 'tch', 'dge', 'ck', 'kn', 'wr', 
            'mb', 'sc', 'ps', 'pn', 'rh', 'mn', 'gn', 'tion', 
            'sion', 'ough', 'augh', 'eigh', 'ieu', 'oux'
        ]
        
        # Medical/technical prefixes and suffixes
        self.medical_affixes = [
            'neuro', 'cardio', 'hemo', 'immuno', 'patho', 'physio',
            'itis', 'osis', 'emia', 'ology', 'ectomy', 'ostomy',
            'mega', 'micro', 'hyper', 'hypo', 'dys', 'mal'
        ]
        
        # Store complexity distribution from training data
        self.complexity_stats = {}
        
    def count_syllables(self, word):
        """Estimate syllable count using a simple algorithm"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
            
        # Ensure at least one syllable
        return max(1, count)
    
    def fit_complexity_distribution(self, words):
        """Learn the distribution of complexity features from training data"""
        all_features = []
        for word in words:
            features = self.analyze_complexity(word)
            all_features.append(features)
        
        # Calculate mean and std for each feature
        feature_names = list(all_features[0].keys())
        for feat in feature_names:
            values = [f[feat] for f in all_features]
            self.complexity_stats[feat] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    def sample_baseline_complexity(self):
        """Sample realistic word complexity for baseline samples"""
        if not self.complexity_stats:
            # Return default if not fitted
            return self.get_default_complexity()
        
        sampled = {}
        for feat, stats in self.complexity_stats.items():
            # Sample from normal distribution clipped to observed range
            value = np.random.normal(stats['mean'], stats['std'])
            value = np.clip(value, stats['min'], stats['max'])
            
            # Convert to appropriate type
            if feat in ['length', 'syllables', 'unique_chars', 'vowel_count', 
                       'consonant_count', 'prefix_count', 'suffix_count']:
                value = int(round(value))
            elif feat in ['has_double_letters', 'has_capital', 'is_medical_term']:
                value = int(np.random.random() < value)
            
            sampled[feat] = value
        
        return sampled
    
    def get_default_complexity(self):
        """Get default complexity values"""
        return {
            'length': 6,
            'syllables': 2,
            'unique_chars': 5,
            'char_variety_ratio': 0.83,
            'vowel_count': 2,
            'consonant_count': 4,
            'vowel_consonant_ratio': 0.5,
            'has_double_letters': 0,
            'has_capital': 0,
            'difficult_pattern_count': 0,
            'is_medical_term': 0,
            'prefix_count': 0,
            'suffix_count': 0,
            'estimated_grade_level': 3.0
        }
    
    def analyze_complexity(self, word):
        """Extract comprehensive complexity features"""
        word_lower = word.lower()
        
        features = {
            # Basic features
            'length': len(word),
            'syllables': self.count_syllables(word),
            
            # Character analysis
            'unique_chars': len(set(word_lower)),
            'char_variety_ratio': len(set(word_lower)) / len(word) if len(word) > 0 else 0,
            
            # Vowel/consonant analysis
            'vowel_count': sum(1 for c in word_lower if c in 'aeiou'),
            'consonant_count': sum(1 for c in word_lower if c.isalpha() and c not in 'aeiou'),
            'vowel_consonant_ratio': 0,  # Will calculate below
            
            # Complexity indicators
            'has_double_letters': int(any(word_lower[i] == word_lower[i+1] 
                                         for i in range(len(word_lower)-1))),
            'has_capital': int(any(c.isupper() for c in word)),
            'difficult_pattern_count': sum(1 for pattern in self.difficult_patterns 
                                         if pattern in word_lower),
            'is_medical_term': int(any(affix in word_lower for affix in self.medical_affixes)),
            
            # Morphological complexity
            'prefix_count': 0,  # Simple heuristic
            'suffix_count': 0,  # Simple heuristic
            
            # Readability estimate (simplified)
            'estimated_grade_level': 0  # Will calculate below
        }
        
        # Calculate vowel/consonant ratio
        if features['consonant_count'] > 0:
            features['vowel_consonant_ratio'] = features['vowel_count'] / features['consonant_count']
        
        # Simple prefix/suffix detection
        common_prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out']
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'ful', 'less']
        
        for prefix in common_prefixes:
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
                features['prefix_count'] += 1
                
        for suffix in common_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                features['suffix_count'] += 1
        
        # Estimate reading grade level (very simplified)
        # Based on syllables and length
        features['estimated_grade_level'] = min(12, features['syllables'] * 1.5 + 
                                               features['length'] * 0.3 + 
                                               features['difficult_pattern_count'] * 2)
        
        return features


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG/fNIRS confusion detection"""
    
    def __init__(self, raw_signals, features, labels, sample_weights=None, augment=False):
        self.raw_signals = raw_signals
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = self.raw_signals['eeg'][idx].copy()
        fnirs = self.raw_signals['fnirs'][idx].copy()
        motion = self.raw_signals['motion'][idx].copy()
        features = self.features[idx].copy()
        label = self.labels[idx]
        
        if self.augment and np.random.rand() > 0.5:
            # Time shift augmentation
            shift = np.random.randint(-5, 5)  # Reduced from -10,10
            eeg = np.roll(eeg, shift, axis=1)
            fnirs = np.roll(fnirs, shift, axis=1)
            
            # Noise augmentation
            if np.random.rand() > 0.5:
                noise_scale = 0.02  # Reduced from 0.05
                eeg += np.random.randn(*eeg.shape) * noise_scale * np.std(eeg)
                fnirs += np.random.randn(*fnirs.shape) * noise_scale * np.std(fnirs)
            
            # Channel dropout
            if np.random.rand() > 0.8:  # Reduced probability
                dropout_ch = np.random.randint(0, eeg.shape[0])
                eeg[dropout_ch] *= 0.5  # Less severe dropout
        
        result = {
            'eeg': torch.FloatTensor(eeg),
            'fnirs': torch.FloatTensor(fnirs),
            'motion': torch.FloatTensor(motion),
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label])[0]
        }
        
        if self.sample_weights is not None:
            result['weight'] = torch.FloatTensor([self.sample_weights[idx]])[0]
        
        return result


class ChannelAttention(nn.Module):
    """Channel attention mechanism for learning channel importance"""
    
    def __init__(self, n_channels):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, n_channels // 2)
        self.fc2 = nn.Linear(n_channels // 2, n_channels)
        self.channel_importance = None  # Store for visualization
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=2)
        attn = F.relu(self.fc1(avg_pool))
        attn = torch.sigmoid(self.fc2(attn))
        self.channel_importance = attn.mean(dim=0).detach()  # Store average importance
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
        self.conv_3 = TemporalConvBlock(n_channels, 32, kernel_size=3)
        self.conv_5 = TemporalConvBlock(n_channels, 32, kernel_size=5)
        self.conv_7 = TemporalConvBlock(n_channels, 32, kernel_size=7)
        self.conv2 = TemporalConvBlock(96, 64, kernel_size=3, stride=2)
        self.conv3 = TemporalConvBlock(64, 128, kernel_size=3, stride=2)
        self.channel_attn = ChannelAttention(n_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.channel_attn(x)
        x1 = self.conv_3(x)
        x2 = self.conv_5(x)
        x3 = self.conv_7(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class fNIRSEncoder(nn.Module):
    """Encoder for fNIRS signals (slower dynamics)"""
    
    def __init__(self, n_channels=8, n_timepoints=512):
        super().__init__()
        # Larger kernels for slower fNIRS dynamics
        self.conv1 = TemporalConvBlock(n_channels, 16, kernel_size=25)  # Increased from 15
        self.conv2 = TemporalConvBlock(16, 32, kernel_size=15, stride=2)  # Increased from 11
        self.conv3 = TemporalConvBlock(32, 64, kernel_size=11, stride=2)  # Increased from 7
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
        
        self.eeg_encoder = MultiScaleEEGEncoder(n_eeg_ch, n_timepoints)
        self.fnirs_encoder = fNIRSEncoder(n_fnirs_ch, n_timepoints)
        
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(n_motion_ch, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
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
        
        fusion_dim = 128 + 64 + 16 + 64  # EEG + fNIRS + motion + features
        
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid()
        )
        
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
        
        # Store feature importance
        self.feature_importance = None
        
    def forward(self, eeg, fnirs, motion, features):
        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)
        motion_feat = self.motion_encoder(motion)
        feat_encoded = self.feature_encoder(features)
        
        fused = torch.cat([eeg_feat, fnirs_feat, motion_feat, feat_encoded], dim=1)
        attn_weights = self.fusion_attention(fused)
        fused = fused * attn_weights
        
        # Store modality importance
        if self.training == False:
            with torch.no_grad():
                self.feature_importance = {
                    'eeg': attn_weights[:, :128].mean().item(),
                    'fnirs': attn_weights[:, 128:192].mean().item(),
                    'motion': attn_weights[:, 192:208].mean().item(),
                    'features': attn_weights[:, 208:].mean().item()
                }
        
        output = self.classifier(fused)
        
        return output, attn_weights


class WordConfusionDetectorNN:
    """Complete neural network pipeline for word-level confusion detection"""
    
    def __init__(self, model_name="confusion_detector_nn"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.complexity_analyzer = WordComplexityAnalyzer()
        self.segment_detector = ReadingSegmentDetector()
        
        self.sample_rate = 256  # Hz for EEG
        self.window_size = 3.0  # Increased from 2.0 seconds
        self.window_samples = int(self.window_size * self.sample_rate)
        
        # Add pre-event offset to account for labeling delay
        self.pre_event_offset = 2.0  # seconds before labeled timestamp
        
        self.history = defaultdict(list)
        
        self.all_data = {
            'timestamps': [],
            'eeg_data': [],
            'fnirs_data': [],
            'motion_data': [],
            'event_timestamps': [],
            'event_types': [],
            'event_words': [],
            'file_indices': []  # Track which file each sample comes from
        }
        
        self.feature_names = None  # Will store feature names for importance viz
        
    def load_multiple_files(self, filepaths):
        """Load and combine data from multiple NPZ files"""
        print(f"\nLoading data from {len(filepaths)} files...")
        
        cumulative_time_offset = 0
        
        for file_idx, filepath in enumerate(filepaths):
            print(f"\nProcessing {filepath}...")
            data = np.load(filepath, allow_pickle=True)
            
            timestamps = data['timestamps']
            n_samples = len(timestamps)
            
            if len(self.all_data['timestamps']) > 0:
                cumulative_time_offset = self.all_data['timestamps'][-1][-1] + 10.0
            
            self.all_data['timestamps'].append(timestamps + cumulative_time_offset)
            self.all_data['eeg_data'].append(data['eeg'])
            self.all_data['fnirs_data'].append(data['fnirs'])
            self.all_data['motion_data'].append(data['motion'])
            self.all_data['file_indices'].extend([file_idx] * n_samples)
            
            # Process events
            if 'event_timestamps' in data and 'event_types' in data and 'event_words' in data:
                self.all_data['event_timestamps'].append(data['event_timestamps'] + cumulative_time_offset)
                self.all_data['event_types'].append(data['event_types'])
                self.all_data['event_words'].append(data['event_words'])
            else:
                print(f"  Warning: No event data found in {filepath}")
                self.all_data['event_timestamps'].append(np.array([]))
                self.all_data['event_types'].append(np.array([]))
                self.all_data['event_words'].append(np.array([]))
            
            print(f"  Loaded {len(timestamps)} samples from {filepath}")
        
        # Concatenate all data
        self.timestamps = np.concatenate(self.all_data['timestamps'])
        self.eeg_data = np.vstack(self.all_data['eeg_data'])
        self.fnirs_data = np.vstack(self.all_data['fnirs_data'])
        self.motion_data = np.vstack(self.all_data['motion_data'])
        self.file_indices = np.array(self.all_data['file_indices'])
        
        # Set timestamps for segment detector
        self.segment_detector.timestamps = self.timestamps
        
        # Handle potentially empty event arrays
        event_arrays = [arr for arr in self.all_data['event_timestamps'] if len(arr) > 0]
        if event_arrays:
            self.event_timestamps = np.concatenate(event_arrays)
            self.event_types = np.concatenate([arr for arr in self.all_data['event_types'] if len(arr) > 0])
            self.event_words = np.concatenate([arr for arr in self.all_data['event_words'] if len(arr) > 0])
        else:
            self.event_timestamps = np.array([])
            self.event_types = np.array([])
            self.event_words = np.array([])
        
        print(f"\nTotal loaded:")
        print(f"  Samples: {len(self.timestamps)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  fNIRS shape: {self.fnirs_data.shape}")
        print(f"  Events: {len(self.event_timestamps)}")
        
    def create_dataset(self):
        """Create training dataset from confusion events with improved baseline selection"""
        print("\nCreating dataset from confusion events...")
        
        if len(self.event_timestamps) == 0:
            raise ValueError("No event data found in the loaded files!")
        
        # Detect reading segments
        print("Detecting reading vs non-reading segments...")
        self.reading_mask = self.segment_detector.detect_reading_segments(
            self.eeg_data, self.motion_data, self.event_timestamps
        )
        
        reading_ratio = np.sum(self.reading_mask) / len(self.reading_mask)
        print(f"Identified {reading_ratio:.1%} of data as active reading")
        
        # Parse confusion events and extract individual words
        confusion_events = []
        all_words = []  # Collect all words for complexity distribution
        
        for i, (ts, event_type, words_str) in enumerate(zip(
            self.event_timestamps, self.event_types, self.event_words)):
            
            # Extract individual words from the event
            if isinstance(words_str, str):
                words = words_str.split()
            else:
                words = [str(words_str)]
            
            all_words.extend([w.lower() for w in words])
            
            # Store each word with its confusion context
            for word in words:
                # Apply pre-event offset to account for labeling delay
                adjusted_ts = ts - self.pre_event_offset
                
                confusion_events.append({
                    'timestamp': adjusted_ts,
                    'original_timestamp': ts,
                    'word': word.lower(),
                    'event_type': event_type,
                    'is_sentence': 'sentence' in str(event_type).lower()
                })
        
        # Fit word complexity distribution
        print(f"Learning word complexity distribution from {len(all_words)} words...")
        self.complexity_analyzer.fit_complexity_distribution(all_words)
        
        print(f"Found {len(confusion_events)} confusion word instances")
        
        # Group confusion events by proximity
        confusion_groups = []
        current_group = []
        
        for event in sorted(confusion_events, key=lambda x: x['timestamp']):
            if not current_group or event['timestamp'] - current_group[-1]['timestamp'] < 3.0:
                current_group.append(event)
            else:
                if current_group:
                    confusion_groups.append(current_group)
                current_group = [event]
        if current_group:
            confusion_groups.append(current_group)
        
        print(f"Grouped into {len(confusion_groups)} confusion episodes")
        
        # Create samples
        samples = []
        raw_signals = {'eeg': [], 'fnirs': [], 'motion': []}
        features = []
        labels = []
        sample_file_indices = []
        sample_weights = []
        
        # Track feature names
        feature_names_list = []
        
        # 1. Create confusion samples
        for group in confusion_groups:
            # Use the timestamp of the first event in the group (after offset adjustment)
            center_event = group[0]  # Use first instead of middle
            ts = center_event['timestamp']
            
            # Find closest sample index
            idx = np.argmin(np.abs(self.timestamps - ts))
            
            # Check if this is during reading
            if not self.reading_mask[idx]:
                continue  # Skip if not during active reading
            
            # Extract window
            half_window = self.window_samples // 2
            start_idx = max(0, idx - half_window)
            end_idx = min(len(self.timestamps), idx + half_window)
            
            if end_idx - start_idx < self.window_samples * 0.8:  # Need at least 80% of window
                continue
            
            # Extract signals
            eeg_win = self.eeg_filtered[start_idx:end_idx].T
            fnirs_win = self.fnirs_filtered[start_idx:end_idx].T
            motion_win = self.motion_data[start_idx:end_idx].T
            
            # Pad or truncate
            eeg_win = self._pad_or_truncate(eeg_win, self.window_samples)
            fnirs_win = self._pad_or_truncate(fnirs_win, self.window_samples)
            motion_win = self._pad_or_truncate(motion_win, self.window_samples)
            
            # Determine label
            if any(event['is_sentence'] for event in group):
                label = 2  # sentence confusion
            else:
                label = 1  # word confusion
            
            # Extract features including word complexity
            word_complexities = []
            for event in group:
                complexity = self.complexity_analyzer.analyze_complexity(event['word'])
                word_complexities.append(complexity)
            
            # Select word with highest estimated grade level
            most_complex_idx = np.argmax([c['estimated_grade_level'] for c in word_complexities])
            word_features = word_complexities[most_complex_idx]
            
            feat_vec, feat_names = self.extract_features(eeg_win, fnirs_win, word_features, return_names=True)
            
            if not feature_names_list:
                feature_names_list = feat_names
            
            raw_signals['eeg'].append(eeg_win)
            raw_signals['fnirs'].append(fnirs_win)
            raw_signals['motion'].append(motion_win)
            features.append(feat_vec)
            labels.append(label)
            sample_file_indices.append(self.file_indices[idx])
            sample_weights.append(1.0)  # Normal weight for confusion samples
        
        # Store feature names
        self.feature_names = feature_names_list
        
        # 2. Create baseline samples with stricter criteria
        n_baseline_needed = int(len(labels) * 1.5)  # Reduced from 2:1 ratio
        baseline_added = 0
        
        print(f"Creating {n_baseline_needed} baseline samples...")
        
        # Find valid baseline regions (active reading, far from events)
        valid_baseline_indices = []
        min_distance_from_events = 10.0  # Increased from 5.0 seconds
        
        for i in range(self.window_samples, len(self.timestamps) - self.window_samples):
            # Must be during active reading
            if not self.reading_mask[i]:
                continue
            
            # Check distance from all confusion events
            ts = self.timestamps[i]
            min_dist = float('inf')
            for event in confusion_events:
                # Use original timestamp for distance check
                min_dist = min(min_dist, abs(ts - event['original_timestamp']))
            
            if min_dist > min_distance_from_events:
                valid_baseline_indices.append(i)
        
        print(f"Found {len(valid_baseline_indices)} valid baseline positions")
        
        # Sample baseline instances
        if len(valid_baseline_indices) > n_baseline_needed:
            baseline_indices = np.random.choice(valid_baseline_indices, n_baseline_needed, replace=False)
        else:
            baseline_indices = valid_baseline_indices
            print(f"Warning: Only {len(baseline_indices)} baseline samples available")
        
        for idx in baseline_indices:
            # Extract window
            half_window = self.window_samples // 2
            start_idx = idx - half_window
            end_idx = idx + half_window
            
            # Extract signals
            eeg_win = self.eeg_filtered[start_idx:end_idx].T
            fnirs_win = self.fnirs_filtered[start_idx:end_idx].T
            motion_win = self.motion_data[start_idx:end_idx].T
            
            # Pad or truncate
            eeg_win = self._pad_or_truncate(eeg_win, self.window_samples)
            fnirs_win = self._pad_or_truncate(fnirs_win, self.window_samples)
            motion_win = self._pad_or_truncate(motion_win, self.window_samples)
            
            # Sample realistic word complexity
            baseline_word_features = self.complexity_analyzer.sample_baseline_complexity()
            
            feat_vec, _ = self.extract_features(eeg_win, fnirs_win, baseline_word_features, return_names=False)
            
            raw_signals['eeg'].append(eeg_win)
            raw_signals['fnirs'].append(fnirs_win)
            raw_signals['motion'].append(motion_win)
            features.append(feat_vec)
            labels.append(0)  # baseline
            sample_file_indices.append(self.file_indices[idx])
            sample_weights.append(1.5)  # Higher weight for baseline to balance
            
            baseline_added += 1
        
        # Convert to arrays
        for key in raw_signals:
            raw_signals[key] = np.array(raw_signals[key])
        features = np.array(features)
        labels = np.array(labels)
        sample_file_indices = np.array(sample_file_indices)
        sample_weights = np.array(sample_weights)
        
        # Store for later use
        self.sample_file_indices = sample_file_indices
        self.sample_weights = sample_weights
        
        # Print class distribution
        print(f"\nClass distribution:")
        label_names = ['baseline', 'word_confusion', 'sentence_confusion']
        for label_val in range(3):
            count = np.sum(labels == label_val)
            if len(labels) > 0:
                print(f"  {label_names[label_val]}: {count} ({count/len(labels)*100:.1f}%)")
        
        # Print file distribution
        print(f"\nSamples per file:")
        for file_idx in range(len(set(self.file_indices))):
            count = np.sum(sample_file_indices == file_idx)
            print(f"  File {file_idx}: {count} samples")
        
        if len(labels) == 0:
            raise ValueError("No valid samples could be created from the data!")
        
        return raw_signals, features, labels
    
    def preprocess_signals(self):
        """Apply signal preprocessing"""
        print("\nPreprocessing signals...")
        
        # EEG preprocessing
        sos = signal.butter(4, [0.5, 40], btype='band', fs=self.sample_rate, output='sos')  # Reduced from 50Hz
        self.eeg_filtered = signal.sosfiltfilt(sos, self.eeg_data, axis=0)
        
        # Notch filters
        for freq in [60, 120]:
            sos_notch = signal.butter(4, [freq-2, freq+2], btype='bandstop', 
                                    fs=self.sample_rate, output='sos')
            self.eeg_filtered = signal.sosfiltfilt(sos_notch, self.eeg_filtered, axis=0)
        
        # fNIRS preprocessing with proper filtering for hemodynamic response
        # Use a low-pass filter appropriate for fNIRS
        sos_fnirs = signal.butter(4, 0.5, btype='low', fs=self.sample_rate, output='sos')
        self.fnirs_filtered = signal.sosfiltfilt(sos_fnirs, self.fnirs_data[:, :4], axis=0)
        self.fnirs_filtered = signal.detrend(self.fnirs_filtered, axis=0)
        
        print("Signal preprocessing complete")
        
    def extract_features(self, eeg_window, fnirs_window, word_complexity, return_names=False):
        """Extract features including word complexity"""
        features = []
        names = []
        
        # EEG features
        eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        for ch_idx, ch_name in enumerate(eeg_channels):
            ch_data = eeg_window[ch_idx]
            
            # Time domain
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(np.abs(ch_data)),
                skew(ch_data),
                kurtosis(ch_data)
            ])
            if return_names:
                names.extend([f'eeg_{ch_name}_mean', f'eeg_{ch_name}_std', 
                             f'eeg_{ch_name}_max_abs', f'eeg_{ch_name}_skew', 
                             f'eeg_{ch_name}_kurtosis'])
            
            # Frequency domain
            freqs, psd = signal.welch(ch_data, fs=self.sample_rate, nperseg=min(256, len(ch_data)))
            
            # Band powers
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)  # Reduced from 50
            }
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features.append(band_power)
                if return_names:
                    names.append(f'eeg_{ch_name}_{band_name}_power')
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)]
            features.append(peak_freq)
            if return_names:
                names.append(f'eeg_{ch_name}_peak_freq')
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-15))
            features.append(spectral_entropy)
            if return_names:
                names.append(f'eeg_{ch_name}_spectral_entropy')
        
        # fNIRS features
        fnirs_channels = ['AF7_O2', 'AF8_O2', 'AF7_HbR', 'AF8_HbR']
        for ch_idx, ch_name in enumerate(fnirs_channels):
            ch_data = fnirs_window[ch_idx]
            
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data) - np.min(ch_data),
                np.polyfit(np.arange(len(ch_data)), ch_data, 1)[0],  # Slope
                np.argmax(ch_data) / len(ch_data)  # Peak timing
            ])
            if return_names:
                names.extend([f'fnirs_{ch_name}_mean', f'fnirs_{ch_name}_std',
                             f'fnirs_{ch_name}_range', f'fnirs_{ch_name}_slope',
                             f'fnirs_{ch_name}_peak_timing'])
        
        # Connectivity features
        af7_alpha = self._get_band_power(eeg_window[1], 'alpha')
        af8_alpha = self._get_band_power(eeg_window[2], 'alpha')
        features.append(af7_alpha - af8_alpha)  # Frontal asymmetry
        if return_names:
            names.append('eeg_frontal_alpha_asymmetry')
        
        # Inter-channel coherence
        channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for ch1, ch2 in channel_pairs:
            coh = self._compute_coherence(eeg_window[ch1], eeg_window[ch2])
            features.append(coh)
            if return_names:
                names.append(f'eeg_coherence_{eeg_channels[ch1]}_{eeg_channels[ch2]}')
        
        # Add word complexity features
        if isinstance(word_complexity, dict):
            complexity_features = [
                word_complexity.get('length', 0),
                word_complexity.get('syllables', 0),
                word_complexity.get('unique_chars', 0),
                word_complexity.get('char_variety_ratio', 0),
                word_complexity.get('vowel_count', 0),
                word_complexity.get('consonant_count', 0),
                word_complexity.get('vowel_consonant_ratio', 0),
                word_complexity.get('has_double_letters', 0),
                word_complexity.get('has_capital', 0),
                word_complexity.get('difficult_pattern_count', 0),
                word_complexity.get('is_medical_term', 0),
                word_complexity.get('prefix_count', 0),
                word_complexity.get('suffix_count', 0),
                word_complexity.get('estimated_grade_level', 0)
            ]
            features.extend(complexity_features)
            if return_names:
                names.extend(['word_length', 'word_syllables', 'word_unique_chars',
                             'word_char_variety_ratio', 'word_vowel_count', 
                             'word_consonant_count', 'word_vowel_consonant_ratio',
                             'word_has_double_letters', 'word_has_capital',
                             'word_difficult_pattern_count', 'word_is_medical_term',
                             'word_prefix_count', 'word_suffix_count',
                             'word_estimated_grade_level'])
        else:
            # Default values if no word complexity provided
            features.extend([0] * 14)
            if return_names:
                names.extend([f'word_feature_{i}' for i in range(14)])
        
        if return_names:
            return np.array(features), names
        return np.array(features)
    
    def _get_band_power(self, signal_data, band_name):
        """Helper to compute band power"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        freqs, psd = signal.welch(signal_data, fs=self.sample_rate, nperseg=min(256, len(signal_data)))
        low, high = bands[band_name]
        band_mask = (freqs >= low) & (freqs < high)
        return np.trapz(psd[band_mask], freqs[band_mask])
    
    def _compute_coherence(self, signal1, signal2):
        """Compute coherence between two signals"""
        f, Cxy = signal.coherence(signal1, signal2, fs=self.sample_rate, 
                                nperseg=min(64, len(signal1)))
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
        """Train the neural network with session-based validation"""
        print("\n" + "="*50)
        print("TRAINING NEURAL NETWORK")
        print("="*50)
        
        # Use session-based split instead of random split
        unique_files = np.unique(self.sample_file_indices)
        n_files = len(unique_files)
        
        if n_files < 2:
            print("Warning: Only one file available, using random split instead of session-based")
            indices = np.arange(len(labels))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, 
                                                stratify=labels, random_state=SEED)
        else:
            # Leave-one-file-out or split files
            val_files = unique_files[-1:]  # Use last file for validation
            train_files = unique_files[:-1]
            
            train_idx = np.isin(self.sample_file_indices, train_files)
            val_idx = np.isin(self.sample_file_indices, val_files)
            
            print(f"Training on files: {train_files}")
            print(f"Validating on files: {val_files}")
        
        # Scale features
        self.scaler.fit(features[train_idx])
        features_scaled = self.scaler.transform(features)
        
        # Create datasets
        train_dataset = EEGDataset(
            {k: v[train_idx] for k, v in raw_signals.items()},
            features_scaled[train_idx],
            labels[train_idx],
            sample_weights=self.sample_weights[train_idx],
            augment=True
        )
        
        val_dataset = EEGDataset(
            {k: v[val_idx] for k, v in raw_signals.items()},
            features_scaled[val_idx],
            labels[val_idx],
            sample_weights=self.sample_weights[val_idx],
            augment=False
        )
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(labels), 
                                           y=labels[train_idx])
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Create weighted sampler
        train_labels = labels[train_idx]
        sample_weights_train = self.sample_weights[train_idx]
        
        # Combine class weights with sample weights
        combined_weights = np.zeros(len(train_labels))
        for i, label in enumerate(train_labels):
            combined_weights[i] = class_weights[label].item() * sample_weights_train[i]
        
        sampler = WeightedRandomSampler(combined_weights, len(combined_weights))
        
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.05)  # Increased regularization
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
        
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
                eeg = batch['eeg'].to(self.device)
                fnirs = batch['fnirs'].to(self.device)
                motion = batch['motion'].to(self.device)
                feat = batch['features'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(eeg, fnirs, motion, feat)
                loss = criterion(outputs, labels_batch)
                
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
            
            scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 25:  # Increased patience
                    print("Early stopping triggered")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pth'))
        
        # Extract feature importance after training
        self._extract_feature_importance(features_scaled, labels)
        
        # Final evaluation
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
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
        
        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)
        
        print("\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            class_mask = all_labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(all_preds[class_mask] == i)
                class_prec = cm[i, i] / (np.sum(cm[:, i]) + 1e-10)
                class_recall = cm[i, i] / (np.sum(cm[i, :]) + 1e-10)
                print(f"  {class_name}: Acc={class_acc:.3f}, "
                      f"Prec={class_prec:.3f}, Recall={class_recall:.3f}")
        
        # Binary confusion detection
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
            print("  AUC: Could not calculate")
        
        return all_preds, all_labels, all_probs
    
    def _extract_feature_importance(self, features, labels):
        """Extract feature importance using gradient-based method"""
        print("\nExtracting feature importance...")
        
        self.model.eval()
        
        # Calculate gradients for each class
        feature_importance_by_class = {0: [], 1: [], 2: []}
        
        for class_idx in range(3):
            class_mask = labels == class_idx
            if np.sum(class_mask) < 10:
                continue
            
            # Sample some instances from this class
            class_features = torch.FloatTensor(features[class_mask][:10]).to(self.device)
            class_features.requires_grad = True
            
            # Create dummy inputs for other modalities
            batch_size = class_features.shape[0]
            dummy_eeg = torch.zeros(batch_size, 4, self.window_samples).to(self.device)
            dummy_fnirs = torch.zeros(batch_size, 8, self.window_samples).to(self.device)
            dummy_motion = torch.zeros(batch_size, 6, self.window_samples).to(self.device)
            
            # Forward pass
            outputs, _ = self.model(dummy_eeg, dummy_fnirs, dummy_motion, class_features)
            
            # Calculate gradients
            target = outputs[:, class_idx].sum()
            target.backward()
            
            # Average absolute gradients
            importance = class_features.grad.abs().mean(dim=0).cpu().numpy()
            feature_importance_by_class[class_idx] = importance
        
        self.feature_importance_by_class = feature_importance_by_class
    
    def visualize_results(self, predictions, labels, probabilities):
        """Create comprehensive visualization of results including feature importance"""
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Training history
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
        
        # Confusion Matrix
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
        
        # ROC Curves
        ax4 = fig.add_subplot(gs[1, 1])
        
        confusion_labels = (labels > 0).astype(int)
        confusion_scores = probabilities[:, 1] + probabilities[:, 2]
        
        try:
            fpr, tpr, _ = roc_curve(confusion_labels, confusion_scores)
            auc_score = auc(fpr, tpr)
            ax4.plot(fpr, tpr, label=f'Confusion Detection (AUC={auc_score:.3f})', 
                    linewidth=2)
        except:
            pass
        
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
        
        # Class Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        class_counts = np.bincount(labels)
        bars = ax5.bar(['Baseline', 'Word', 'Sentence'], class_counts, 
                       color=['#2ecc71', '#e74c3c', '#f39c12'])
        
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax5.set_ylabel('Count')
        ax5.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax5.grid(True, axis='y', alpha=0.3)
        
        # Feature Importance - Top features for confusion detection
        ax6 = fig.add_subplot(gs[2, :])
        
        if hasattr(self, 'feature_importance_by_class') and self.feature_names:
            # Average importance across confusion classes (1 and 2)
            confusion_importance = []
            for class_idx in [1, 2]:
                if class_idx in self.feature_importance_by_class:
                    if len(confusion_importance) == 0:
                        confusion_importance = self.feature_importance_by_class[class_idx]
                    else:
                        confusion_importance = (confusion_importance + 
                                              self.feature_importance_by_class[class_idx]) / 2
            
            if len(confusion_importance) > 0:
                # Get top 20 features
                top_indices = np.argsort(confusion_importance)[-20:][::-1]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = confusion_importance[top_indices]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_features))
                ax6.barh(y_pos, top_importance, color='#3498db')
                ax6.set_yticks(y_pos)
                ax6.set_yticklabels(top_features, fontsize=10)
                ax6.set_xlabel('Feature Importance (Gradient Magnitude)')
                ax6.set_title('Top 20 Features for Confusion Detection', fontsize=14, fontweight='bold')
                ax6.grid(True, axis='x', alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        # Modality Importance
        ax7 = fig.add_subplot(gs[3, 0])
        
        if hasattr(self.model, 'feature_importance') and self.model.feature_importance:
            modalities = list(self.model.feature_importance.keys())
            importance_values = list(self.model.feature_importance.values())
            
            bars = ax7.bar(modalities, importance_values, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
            ax7.set_ylabel('Attention Weight')
            ax7.set_title('Modality Importance (Fusion Attention)', fontsize=14, fontweight='bold')
            ax7.grid(True, axis='y', alpha=0.3)
            
            for bar, val in zip(bars, importance_values):
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        else:
            ax7.text(0.5, 0.5, 'Modality importance not available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Modality Importance', fontsize=14, fontweight='bold')
        
        # Channel Importance (EEG)
        ax8 = fig.add_subplot(gs[3, 1])
        
        if hasattr(self.model.eeg_encoder.channel_attn, 'channel_importance'):
            channel_imp = self.model.eeg_encoder.channel_attn.channel_importance
            if channel_imp is not None:
                channels = ['TP9', 'AF7', 'AF8', 'TP10']
                channel_values = channel_imp.cpu().numpy()
                
                bars = ax8.bar(channels, channel_values, color='#9b59b6')
                ax8.set_ylabel('Channel Weight')
                ax8.set_title('EEG Channel Importance', fontsize=14, fontweight='bold')
                ax8.grid(True, axis='y', alpha=0.3)
                
                for bar, val in zip(bars, channel_values):
                    ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom')
        else:
            ax8.text(0.5, 0.5, 'Channel importance not available', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('EEG Channel Importance', fontsize=14, fontweight='bold')
        
        # Confidence Distribution
        ax9 = fig.add_subplot(gs[3, 2])
        
        confidence_scores = []
        for i, pred in enumerate(predictions):
            confidence_scores.append(probabilities[i, pred])
        
        for class_idx, class_name in enumerate(['Baseline', 'Word Confusion', 'Sentence Confusion']):
            mask = labels == class_idx
            if np.sum(mask) > 0:
                class_confidences = [confidence_scores[i] for i in range(len(labels)) if mask[i]]
                ax9.hist(class_confidences, bins=20, alpha=0.6, label=class_name, 
                        density=True)
        
        ax9.set_xlabel('Prediction Confidence')
        ax9.set_ylabel('Density')
        ax9.set_title('Prediction Confidence Distribution by True Class', 
                     fontsize=14, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Performance Summary
        ax10 = fig.add_subplot(gs[4, :])
        ax10.axis('off')
        
        overall_acc = np.mean(predictions == labels)
        
        metrics_text = "PERFORMANCE SUMMARY\n" + "="*60 + "\n\n"
        metrics_text += f"Overall Accuracy: {overall_acc:.3f}\n\n"
        
        # Add data quality metrics
        if hasattr(self, 'reading_mask'):
            reading_ratio = np.sum(self.reading_mask) / len(self.reading_mask)
            metrics_text += f"Active Reading Time: {reading_ratio:.1%}\n"
        
        metrics_text += f"Pre-event Offset: {self.pre_event_offset}s\n"
        metrics_text += f"Window Size: {self.window_size}s\n\n"
        
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
        
        confusion_binary = (labels > 0).astype(int)
        confusion_pred_binary = (predictions > 0).astype(int)
        cm_binary = confusion_matrix(confusion_binary, confusion_pred_binary)
        
        metrics_text += "Binary Confusion Detection:\n"
        if np.sum(cm_binary[1,:]) > 0:
            metrics_text += f"  Sensitivity: {cm_binary[1,1]/np.sum(cm_binary[1,:]):.3f}\n"
        if np.sum(cm_binary[0,:]) > 0:
            metrics_text += f"  Specificity: {cm_binary[0,0]/np.sum(cm_binary[0,:]):.3f}\n"
        
        # Add warnings about potential issues
        metrics_text += "\n" + "="*60 + "\n"
        metrics_text += "DATA QUALITY CHECKS:\n"
        
        if overall_acc > 0.85:
            metrics_text += "⚠️  WARNING: High accuracy may indicate data leakage\n"
        
        if hasattr(self, 'sample_file_indices'):
            files_in_train = len(np.unique(self.sample_file_indices))
            if files_in_train < 3:
                metrics_text += f"⚠️  WARNING: Only {files_in_train} files - limited generalization\n"
        
        ax10.text(0.05, 0.95, metrics_text, transform=ax10.transAxes, 
                fontfamily='monospace', fontsize=11, verticalalignment='top')
        
        plt.suptitle('EEG/fNIRS Confusion Detection - Neural Network Results (Fixed)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.model_name}_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, filepath=None):
        """Save the complete model package"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'{self.model_name}_{timestamp}.pth'
        
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
            'complexity_analyzer': self.complexity_analyzer,
            'history': dict(self.history),
            'sample_rate': self.sample_rate,
            'window_size': self.window_size,
            'pre_event_offset': self.pre_event_offset,
            'feature_names': self.feature_names,
            'feature_importance_by_class': getattr(self, 'feature_importance_by_class', None),
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'model_type': 'neural_network_with_complexity_fixed',
                'framework': 'pytorch',
                'version': '3.0'
            }
        }
        
        torch.save(model_package, filepath)
        print(f"\nModel saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Train neural network for EEG/fNIRS word confusion detection (Fixed version)'
    )
    parser.add_argument(
        'data_files',
        nargs='+',
        help='Path(s) to NPZ data files'
    )
    parser.add_argument(
        '--model-name',
        default='confusion_detector_nn_fixed',
        help='Name for the model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save outputs'
    )
    
    args = parser.parse_args()
    
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
    
    # Load data
    detector.load_multiple_files(data_files)
    
    # Preprocess
    detector.preprocess_signals()
    
    # Create dataset
    try:
        raw_signals, features, labels = detector.create_dataset()
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Train
    predictions, true_labels, probabilities = detector.train_model(
        raw_signals, features, labels, 
        n_epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Visualize
    detector.visualize_results(predictions, true_labels, probabilities)
    
    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = detector.save_model(
        str(output_path / f"{args.model_name}_trained.pth")
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()