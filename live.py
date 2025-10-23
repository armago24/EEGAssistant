#!/usr/bin/env python3
"""
live.py - Real-time Confusion Detection System
Combines EEG/fNIRS data collection with neural network inference
to predict reading confusion in real-time.
"""

import socket
import struct
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import os
import datetime
import scipy.signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

# ===========================
# NEURAL NETWORK ARCHITECTURE (Matching neural.py exactly)
# ===========================

class ChannelAttention(nn.Module):
    """Channel attention mechanism for learning channel importance"""
    
    def __init__(self, n_channels):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, n_channels // 2)
        self.fc2 = nn.Linear(n_channels // 2, n_channels)
        self.channel_importance = None
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=2)
        attn = F.relu(self.fc1(avg_pool))
        attn = torch.sigmoid(self.fc2(attn))
        self.channel_importance = attn.mean(dim=0).detach()
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
        self.conv1 = TemporalConvBlock(n_channels, 16, kernel_size=25)
        self.conv2 = TemporalConvBlock(16, 32, kernel_size=15, stride=2)
        self.conv3 = TemporalConvBlock(32, 64, kernel_size=11, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class ConfusionDetectorNN(nn.Module):
    """Main neural network for confusion detection - exact match of neural.py"""
    
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
        
        self.feature_importance = None
        
    def forward(self, eeg, fnirs, motion, features):
        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)
        motion_feat = self.motion_encoder(motion)
        feat_encoded = self.feature_encoder(features)
        
        fused = torch.cat([eeg_feat, fnirs_feat, motion_feat, feat_encoded], dim=1)
        attn_weights = self.fusion_attention(fused)
        fused = fused * attn_weights
        
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


# ===========================
# WORD COMPLEXITY ANALYZER (Matching neural.py)
# ===========================

class WordComplexityAnalyzer:
    """Analyze word complexity features"""
    
    def __init__(self):
        self.difficult_patterns = [
            'ph', 'gh', 'ght', 'tch', 'dge', 'ck', 'kn', 'wr', 
            'mb', 'sc', 'ps', 'pn', 'rh', 'mn', 'gn', 'tion', 
            'sion', 'ough', 'augh', 'eigh', 'ieu', 'oux'
        ]
        
        self.medical_affixes = [
            'neuro', 'cardio', 'hemo', 'immuno', 'patho', 'physio',
            'itis', 'osis', 'emia', 'ology', 'ectomy', 'ostomy',
            'mega', 'micro', 'hyper', 'hypo', 'dys', 'mal'
        ]
        
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
        
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
            
        return max(1, count)
    
    def analyze_complexity(self, word):
        """Extract comprehensive complexity features"""
        word_lower = word.lower()
        
        features = {
            'length': len(word),
            'syllables': self.count_syllables(word),
            'unique_chars': len(set(word_lower)),
            'char_variety_ratio': len(set(word_lower)) / len(word) if len(word) > 0 else 0,
            'vowel_count': sum(1 for c in word_lower if c in 'aeiou'),
            'consonant_count': sum(1 for c in word_lower if c.isalpha() and c not in 'aeiou'),
            'vowel_consonant_ratio': 0,
            'has_double_letters': int(any(word_lower[i] == word_lower[i+1] 
                                         for i in range(len(word_lower)-1))),
            'has_capital': int(any(c.isupper() for c in word)),
            'difficult_pattern_count': sum(1 for pattern in self.difficult_patterns 
                                         if pattern in word_lower),
            'is_medical_term': int(any(affix in word_lower for affix in self.medical_affixes)),
            'prefix_count': 0,
            'suffix_count': 0,
            'estimated_grade_level': 0
        }
        
        if features['consonant_count'] > 0:
            features['vowel_consonant_ratio'] = features['vowel_count'] / features['consonant_count']
        
        common_prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out']
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'ful', 'less']
        
        for prefix in common_prefixes:
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
                features['prefix_count'] += 1
                
        for suffix in common_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                features['suffix_count'] += 1
        
        features['estimated_grade_level'] = min(12, features['syllables'] * 1.5 + 
                                               features['length'] * 0.3 + 
                                               features['difficult_pattern_count'] * 2)
        
        return features


# ===========================
# REAL-TIME CONFUSION DETECTOR
# ===========================
#
# TEMPORAL ALIGNMENT:
# -------------------
# Training uses centered 3-second windows around confusion events:
#   Window = [T_confusion - 1.5s, T_confusion + 1.5s]
#
# Live deployment must match this:
# 1. User reads word at T_read -> Store (word, T_read, position)
# 2. Wait 1.5 seconds (half of window_duration)
# 3. At T_read + 1.5s, current 3s buffer contains [T_now - 3.0, T_now]
# 4. Since T_now = T_read + 1.5, buffer = [T_read - 1.5, T_read + 1.5]
# 5. This matches training: centered window around reading time
# 6. Make prediction using entire current buffer
# 7. Highlight if confidence > threshold
#
# Note: fNIRS has 4-6s hemodynamic lag, but training captured rising phase,
# not peak. Live deployment must use same timing to match trained model.
# ===========================

class RealTimeConfusionDetector:
    def __init__(self, model_path=None):
        # Neural network components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False

        # Data buffers - using 3 second windows to match model training
        self.sample_rate = 256
        self.window_duration = 3.0  # seconds (matches neural.py)
        self.window_size = int(self.window_duration * self.sample_rate)  # 768 samples
        self.eeg_buffer = deque(maxlen=self.window_size)
        self.fnirs_buffer = deque(maxlen=self.window_size)
        self.motion_buffer = deque(maxlen=self.window_size)

        # Buffer warmup tracking - CRITICAL FIX
        self.buffer_ready = False
        self.first_data_time = None
        self.warmup_duration = 3.5  # Need 3.5s of real data before predictions

        # Word tracking with timing
        self.current_word = ""
        self.word_history = deque(maxlen=100)  # Store more history for delayed prediction
        self.word_reading_history = deque(maxlen=100)  # (word, timestamp, text_position)
        self.word_predictions = {}
        self.complexity_analyzer = WordComplexityAnalyzer()

        # Timing parameters: Must match training temporal alignment
        # Training: window centered at (click_time - 2.0s) with span of 3.0s
        # This gives [click - 3.5s, click - 0.5s]
        # Live: Need window centered at reading_time with same 3.0s span
        # After waiting 1.5s, buffer = [T_read - 1.5s, T_read + 1.5s] = centered window
        self.pre_event_offset = 1.5  # Half of window_duration - critical for temporal alignment

        # Reading state detection
        self.last_word_change_time = time.time()
        self.reading_active = False
        self.cursor_stationary_threshold = 1.0  # seconds
        self.words_per_minute_threshold = 50  # minimum reading speed

        # Confusion tracking - store (word, position) tuples to prevent pre-reading highlights
        self.confused_word_positions = set()  # Set of (word, text_index) tuples
        self.confusion_threshold = 0.7  # Increased from 0.5 - be conservative to reduce false positives

        # Prediction quality tracking
        self.recent_predictions = deque(maxlen=50)  # Track recent predictions for sanity checks
        self.prediction_count = 0
        self.high_confidence_count = 0

        # Temporal smoothing for predictions
        self.word_prediction_history = {}  # word_index -> list of (timestamp, probability)
        self.prediction_smoothing_window = 3.0  # seconds

        # DO NOT pre-fill buffers with zeros - this causes OOD predictions!
        # Buffers will fill naturally as data arrives

        # Feature extraction parameters
        self.sample_rate = 256
        self.feature_names = None

        # Thread safety
        self.lock = threading.Lock()

        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load pre-trained model and associated components

        IMPORTANT: Model was trained with centered 3s windows around confusion events.
        Live deployment must use the same window timing: wait 1.5s after reading,
        then use the current 3s buffer which will be centered on the reading time.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model architecture using config from checkpoint
            model_config = checkpoint.get('model_config', {})
            self.model = ConfusionDetectorNN(
                n_eeg_ch=model_config.get('n_eeg_ch', 4),
                n_fnirs_ch=model_config.get('n_fnirs_ch', 8),
                n_motion_ch=model_config.get('n_motion_ch', 6),
                n_features=model_config.get('n_features', 100),
                n_classes=model_config.get('n_classes', 3),
                n_timepoints=model_config.get('n_timepoints', 512)
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            self.scaler = checkpoint.get('scaler')
            
            # Load complexity analyzer
            if 'complexity_analyzer' in checkpoint:
                self.complexity_analyzer = checkpoint['complexity_analyzer']
            
            # Load feature names if available
            self.feature_names = checkpoint.get('feature_names', None)
            
            # Load window parameters from checkpoint
            checkpoint_window_size = checkpoint.get('window_size', 3.0)
            checkpoint_sample_rate = checkpoint.get('sample_rate', 256)
            checkpoint_window_samples = int(checkpoint_window_size * checkpoint_sample_rate)

            # Update our parameters to match
            self.window_duration = checkpoint_window_size
            self.sample_rate = checkpoint_sample_rate
            self.window_size = checkpoint_window_samples

            # Resize buffers if needed
            if len(self.eeg_buffer) != self.window_size:
                self.eeg_buffer = deque(maxlen=self.window_size)
                self.fnirs_buffer = deque(maxlen=self.window_size)
                self.motion_buffer = deque(maxlen=self.window_size)

                # Initialize with zeros
                for _ in range(self.window_size):
                    self.eeg_buffer.append(np.zeros(4))
                    self.fnirs_buffer.append(np.zeros(8))
                    self.motion_buffer.append(np.zeros(6))
            
            self.model_loaded = True
            print(f"✓ Model loaded successfully from {model_path}")
            print(f"  Model expects {model_config.get('n_features', '?')} features")

            # CRITICAL: Validate feature extraction matches training
            self._validate_feature_extraction(checkpoint)

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def _validate_feature_extraction(self, checkpoint):
        """Validate that live feature extraction matches training"""
        try:
            print("\nValidating feature extraction compatibility...")

            # Create dummy data
            dummy_eeg = np.random.randn(self.window_size, 4)
            dummy_fnirs = np.random.randn(self.window_size, 8)
            dummy_word = "test"

            # Preprocess
            eeg_filtered, fnirs_filtered = self.preprocess_signals(dummy_eeg, dummy_fnirs)

            # Extract features
            word_features = self.complexity_analyzer.analyze_complexity(dummy_word)
            extracted_features = self.extract_features(eeg_filtered.T, fnirs_filtered.T, word_features)

            # Check dimensions
            expected_features = checkpoint.get('model_config', {}).get('n_features', None)

            if expected_features is None:
                print("  WARNING: Model config missing n_features - cannot validate!")
                print(f"  Live extraction produces {len(extracted_features)} features")
            elif len(extracted_features) != expected_features:
                raise ValueError(
                    f"Feature dimension MISMATCH!\n"
                    f"  Model expects: {expected_features} features\n"
                    f"  Live extracts: {len(extracted_features)} features\n"
                    f"  This will cause prediction errors!"
                )
            else:
                print(f"  ✓ Feature dimensions match: {len(extracted_features)} features")

            # Validate feature names if available
            if 'feature_names' in checkpoint and checkpoint['feature_names']:
                stored_names = checkpoint['feature_names']
                if len(stored_names) != len(extracted_features):
                    print(f"  WARNING: Feature count changed ({len(stored_names)} -> {len(extracted_features)})")

            print("  ✓ Feature extraction validation complete\n")

        except Exception as e:
            print(f"  WARNING: Feature validation failed: {e}")
            print("  Proceeding anyway, but predictions may be incorrect!\n")
    
    def add_eeg_sample(self, eeg_data):
        """Add EEG sample to buffer thread-safely"""
        if len(eeg_data) == 4:
            with self.lock:
                # Track first data arrival for warmup
                if self.first_data_time is None:
                    self.first_data_time = time.time()
                    print(f"Data collection started - warming up for {self.warmup_duration}s...")

                self.eeg_buffer.append(np.array(eeg_data))

                # Check if warmup period complete
                if not self.buffer_ready:
                    if time.time() - self.first_data_time >= self.warmup_duration:
                        self.buffer_ready = True
                        print(f"Buffer ready! Warmup complete ({len(self.eeg_buffer)} samples)")

    def add_fnirs_sample(self, fnirs_data):
        """Add fNIRS sample to buffer thread-safely"""
        if len(fnirs_data) == 8:
            with self.lock:
                self.fnirs_buffer.append(np.array(fnirs_data))

    def add_motion_sample(self, motion_data):
        """Add motion sample to buffer thread-safely"""
        if len(motion_data) == 6:
            with self.lock:
                self.motion_buffer.append(np.array(motion_data))

    def detect_reading_state(self):
        """Detect if user is actively reading based on word change patterns"""
        current_time = time.time()
        time_since_last_change = current_time - self.last_word_change_time

        # Check if cursor has been stationary too long
        if time_since_last_change > self.cursor_stationary_threshold:
            self.reading_active = False
            return False

        # Check reading speed (words per minute)
        if len(self.word_history) >= 5:
            # Calculate WPM from recent word changes
            recent_words = list(self.word_history)[-5:]
            if len(set(recent_words)) >= 3:  # At least 3 different words
                # Estimate reading is active if we have word variety
                self.reading_active = True
                return True

        # Default to not reading if uncertain
        self.reading_active = False
        return False

    def update_word_reading_history(self, word, text_position):
        """Update word reading history with timestamp and position"""
        if not word or word == self.current_word:
            return

        current_time = time.time()
        self.last_word_change_time = current_time

        # Store word with timestamp and text position (no buffer snapshot)
        self.word_reading_history.append({
            'word': word,
            'timestamp': current_time,
            'text_position': text_position
        })

        self.current_word = word
        self.word_history.append(word)

        # Update reading state
        self.detect_reading_state()

    def get_delayed_prediction_candidates(self):
        """Get words that should be predicted after sufficient delay for window completion"""
        current_time = time.time()
        candidates = []

        for entry in self.word_reading_history:
            time_diff = current_time - entry['timestamp']
            # Check if this word was read approximately pre_event_offset seconds ago
            # Allow window of +/- 0.2 seconds for matching
            if self.pre_event_offset - 0.2 <= time_diff <= self.pre_event_offset + 0.3:
                # Check if we haven't already predicted this word
                word_key = (entry['word'], entry['timestamp'])
                if word_key not in self.word_predictions:
                    candidates.append(entry)

        return candidates

    def process_delayed_predictions(self):
        """Process predictions for words using current buffer state after delay"""
        if not self.model_loaded or not self.buffer_ready:
            return

        candidates = self.get_delayed_prediction_candidates()

        for entry in candidates:
            # Verify entry has all required fields
            if not all(key in entry for key in ['word', 'timestamp', 'text_position']):
                continue

            word = entry['word']
            timestamp = entry['timestamp']
            text_position = entry['text_position']

            try:
                # Get CURRENT buffer state (after 1.5s delay has elapsed)
                # Buffer contains last 3.0 seconds: [T_now - 3.0, T_now]
                # Since T_now = T_read + 1.5, buffer = [T_read - 1.5, T_read + 1.5]
                # This matches training: centered 3s window around confusion time
                with self.lock:
                    if len(self.eeg_buffer) < self.window_size:
                        continue

                    eeg_snapshot = np.vstack(list(self.eeg_buffer)).copy()
                    fnirs_snapshot = np.vstack(list(self.fnirs_buffer)).copy()
                    motion_snapshot = np.vstack(list(self.motion_buffer)).copy()

                # Preprocess the signals
                eeg_filtered, fnirs_filtered = self.preprocess_signals(eeg_snapshot, fnirs_snapshot)

                # Extract word complexity features
                word_features = self.complexity_analyzer.analyze_complexity(word)

                # Extract ALL features
                all_features = self.extract_features(eeg_filtered.T, fnirs_filtered.T, word_features)

                # Scale features if scaler available
                if self.scaler:
                    all_features_scaled = self.scaler.transform(all_features.reshape(1, -1))
                else:
                    all_features_scaled = all_features.reshape(1, -1)

                # Prepare tensors
                eeg_tensor = torch.FloatTensor(eeg_filtered.T.copy()).unsqueeze(0).to(self.device)
                fnirs_tensor = torch.FloatTensor(fnirs_filtered.T.copy()).unsqueeze(0).to(self.device)
                motion_tensor = torch.FloatTensor(motion_snapshot.T.copy()).unsqueeze(0).to(self.device)
                features_tensor = torch.FloatTensor(all_features_scaled).to(self.device)

                # Predict
                with torch.no_grad():
                    output, _ = self.model(eeg_tensor, fnirs_tensor, motion_tensor, features_tensor)
                    probs = F.softmax(output, dim=1)
                    confusion_prob = float(probs[0, 1] + probs[0, 2])

                # Track prediction quality
                self.prediction_count += 1
                if confusion_prob > 0.9:
                    self.high_confidence_count += 1
                self.recent_predictions.append(confusion_prob)

                # Sanity check: warn if model predicts everything as confused
                if self.prediction_count >= 20:
                    high_conf_ratio = self.high_confidence_count / self.prediction_count
                    if high_conf_ratio > 0.8:
                        print(f"\n⚠️  WARNING: Model predicting {high_conf_ratio:.0%} of words as highly confused!")
                        print(f"   This suggests temporal misalignment or model calibration issues.")
                        print(f"   Recent predictions: {[f'{p:.2f}' for p in list(self.recent_predictions)[-10:]]}\n")

                # Store prediction with timestamp and position
                word_key = (word, timestamp)
                self.word_predictions[word_key] = {
                    'probability': confusion_prob,
                    'prediction_time': time.time(),
                    'word': word,
                    'read_time': timestamp,
                    'text_position': text_position
                }

                # Debug output with signal quality indicators
                if confusion_prob > 0.4:
                    # Calculate signal quality metrics
                    eeg_std = np.std(eeg_snapshot)
                    fnirs_std = np.std(fnirs_snapshot)
                    buffer_fullness = len(self.eeg_buffer) / self.window_size

                    print(f"PREDICTED '{word}' at {text_position} | "
                          f"prob={confusion_prob:.3f} | "
                          f"buffer={buffer_fullness:.0%} | "
                          f"eeg_std={eeg_std:.2f} | "
                          f"fnirs_std={fnirs_std:.2f}")

            except Exception as e:
                print(f"Delayed prediction error for word '{word}': {e}")
                import traceback
                traceback.print_exc()

    def get_smoothed_confusion_probability(self, word, current_time):
        """Get temporally smoothed confusion probability for a word"""
        # Find all recent predictions for this word
        recent_probs = []

        for (pred_word, pred_timestamp), pred_data in self.word_predictions.items():
            if pred_word.lower() == word.lower():
                time_diff = current_time - pred_data['prediction_time']
                if time_diff <= self.prediction_smoothing_window:
                    # Weight by recency (more recent = higher weight)
                    weight = 1.0 - (time_diff / self.prediction_smoothing_window)
                    recent_probs.append((pred_data['probability'], weight))

        if not recent_probs:
            return 0.0

        # Weighted average
        total_weight = sum(w for _, w in recent_probs)
        if total_weight == 0:
            return 0.0

        smoothed_prob = sum(p * w for p, w in recent_probs) / total_weight
        return smoothed_prob
    
    def preprocess_signals(self, eeg, fnirs):
        """Apply preprocessing to signals - matching neural.py"""
        # EEG: Bandpass filter 0.5-40 Hz
        sos = scipy.signal.butter(4, [0.5, 40], btype='band', fs=self.sample_rate, output='sos')
        eeg_filtered = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
        
        # Notch filters at 60 and 120 Hz
        for freq in [60, 120]:
            notch_sos = scipy.signal.butter(4, [freq-2, freq+2], btype='bandstop', 
                                          fs=self.sample_rate, output='sos')
            eeg_filtered = scipy.signal.sosfiltfilt(notch_sos, eeg_filtered, axis=0)
        
        # fNIRS: Lowpass filter
        sos_fnirs = scipy.signal.butter(4, 0.5, btype='low', fs=self.sample_rate, output='sos')
        fnirs_filtered = scipy.signal.sosfiltfilt(sos_fnirs, fnirs, axis=0)
        fnirs_filtered = scipy.signal.detrend(fnirs_filtered, axis=0)
        
        return eeg_filtered, fnirs_filtered
    
    def extract_features(self, eeg_window, fnirs_window, word_features):
        """Extract features matching neural.py format exactly"""
        features = []
        
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
            
            # Frequency domain
            freqs, psd = scipy.signal.welch(ch_data, fs=self.sample_rate, 
                                           nperseg=min(256, len(ch_data)))
            
            # Band powers
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)
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
        
        # fNIRS features
        n_fnirs_channels = fnirs_window.shape[0]
        if n_fnirs_channels >= 8:
            fnirs_channels = ['AF7_O2', 'AF8_O2', 'AF7_HbR', 'AF8_HbR', 
                            'FP1_O2', 'FP2_O2', 'FP1_HbR', 'FP2_HbR'][:n_fnirs_channels]
        else:
            fnirs_channels = [f'fnirs_ch{i}' for i in range(n_fnirs_channels)]
        
        for ch_idx, ch_name in enumerate(fnirs_channels):
            ch_data = fnirs_window[ch_idx]
            
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data) - np.min(ch_data),
                np.polyfit(np.arange(len(ch_data)), ch_data, 1)[0],
                np.argmax(ch_data) / len(ch_data)
            ])
        
        # Connectivity features
        af7_alpha = self._get_band_power(eeg_window[1], 'alpha')
        af8_alpha = self._get_band_power(eeg_window[2], 'alpha')
        features.append(af7_alpha - af8_alpha)
        
        # Inter-channel coherence
        channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for ch1, ch2 in channel_pairs:
            coh = self._compute_coherence(eeg_window[ch1], eeg_window[ch2])
            features.append(coh)
        
        # Word complexity features
        if isinstance(word_features, dict):
            complexity_features = [
                word_features.get('length', 0),
                word_features.get('syllables', 0),
                word_features.get('unique_chars', 0),
                word_features.get('char_variety_ratio', 0),
                word_features.get('vowel_count', 0),
                word_features.get('consonant_count', 0),
                word_features.get('vowel_consonant_ratio', 0),
                word_features.get('has_double_letters', 0),
                word_features.get('has_capital', 0),
                word_features.get('difficult_pattern_count', 0),
                word_features.get('is_medical_term', 0),
                word_features.get('prefix_count', 0),
                word_features.get('suffix_count', 0),
                word_features.get('estimated_grade_level', 0)
            ]
            features.extend(complexity_features)
        else:
            features.extend([0] * 14)
        
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
        
        freqs, psd = scipy.signal.welch(signal_data, fs=self.sample_rate, 
                                       nperseg=min(256, len(signal_data)))
        low, high = bands[band_name]
        band_mask = (freqs >= low) & (freqs < high)
        return np.trapz(psd[band_mask], freqs[band_mask])
    
    def _compute_coherence(self, signal1, signal2):
        """Compute coherence between two signals"""
        f, Cxy = scipy.signal.coherence(signal1, signal2, fs=self.sample_rate, 
                                      nperseg=min(64, len(signal1)))
        alpha_mask = (f >= 8) & (f <= 13)
        return np.mean(Cxy[alpha_mask])
    
    def predict_confusion(self, word):
        """Predict if current word causes confusion"""
        if not self.model_loaded or not word:
            return 0.0, "no_model"

        try:
            # Get current data window with thread safety
            with self.lock:
                if len(self.eeg_buffer) < self.window_size:
                    return 0.0, "insufficient_data"

                # Convert buffer contents to numpy arrays
                eeg = np.vstack(list(self.eeg_buffer))  # Shape: (window_size, 4)
                fnirs = np.vstack(list(self.fnirs_buffer))  # Shape: (window_size, 8)
                motion = np.vstack(list(self.motion_buffer))  # Shape: (window_size, 6)

            # Preprocess
            eeg, fnirs = self.preprocess_signals(eeg, fnirs)

            # Extract word complexity features
            word_features = self.complexity_analyzer.analyze_complexity(word)

            # Extract ALL features (not just word features!)
            all_features = self.extract_features(eeg.T, fnirs.T, word_features)

            # Scale features if scaler available
            if self.scaler:
                all_features_scaled = self.scaler.transform(all_features.reshape(1, -1))
            else:
                all_features_scaled = all_features.reshape(1, -1)

            # Prepare tensors
            eeg_tensor = torch.FloatTensor(eeg.T.copy()).unsqueeze(0).to(self.device)
            fnirs_tensor = torch.FloatTensor(fnirs.T.copy()).unsqueeze(0).to(self.device)
            motion_tensor = torch.FloatTensor(motion.T.copy()).unsqueeze(0).to(self.device)

            # Use the FULL scaled feature vector
            features_tensor = torch.FloatTensor(all_features_scaled).to(self.device)

            # Predict
            with torch.no_grad():
                output, _ = self.model(eeg_tensor, fnirs_tensor, motion_tensor, features_tensor)
                probs = F.softmax(output, dim=1)

                # Get confusion probability (word + sentence confusion)
                confusion_prob = probs[0, 1] + probs[0, 2]

            pred_class = output.argmax(dim=1).item()
            class_names = ['baseline', 'word_confusion', 'sentence_confusion']

            return float(confusion_prob), class_names[pred_class]

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, "error"


# ===========================
# OSC DATA RECEIVER
# ===========================

class OSCReceiver:
    def __init__(self, port=8052):
        self.port = port
        self.sock = None
        self.callbacks = {
            '/muse/eeg': None,
            '/muse/optics': None,
            '/muse/acc': None,
            '/muse/gyro': None,
            '/muse/drlref': None
        }
        self.running = False
        self.packet_count = 0

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.port))
        self.sock.settimeout(0.1)
        self.running = True

        receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        receiver_thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

    def set_callback(self, address, callback):
        self.callbacks[address] = callback

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.packet_count += 1
                message = self._parse_osc_message(data)
                if message:
                    self._process_osc_message(message)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")

    def _parse_osc_message(self, data):
        """Parse OSC message from binary data - robust version from eegTrainer.py"""
        try:
            def parse_string(data, offset):
                end = data.find(b'\x00', offset)
                if end == -1:
                    return None, offset
                string = data[offset:end].decode('ascii')
                offset = ((end + 4) // 4) * 4
                return string, offset

            offset = 0
            address, offset = parse_string(data, offset)
            if not address:
                return None

            if not address.startswith('/'):
                address = '/' + address

            type_tags, offset = parse_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None

            type_tags = type_tags[1:]
            args = []

            for tag in type_tags:
                if tag == 'f' and offset + 4 <= len(data):
                    args.append(struct.unpack('>f', data[offset:offset+4])[0])
                    offset += 4
                elif tag == 'i' and offset + 4 <= len(data):
                    args.append(struct.unpack('>i', data[offset:offset+4])[0])
                    offset += 4

            return {'address': address, 'args': args}
        except:
            return None

    def _process_osc_message(self, message):
        """Process incoming OSC message"""
        address = message['address']
        args = message['args']

        parts = address.strip('/').split('/')
        if len(parts) >= 2:
            data_type = parts[1]

            # Map to callback addresses
            callback_map = {
                'eeg': '/muse/eeg',
                'optics': '/muse/optics',
                'acc': '/muse/acc',
                'gyro': '/muse/gyro',
                'drlref': '/muse/drlref'
            }

            if data_type in callback_map:
                callback_addr = callback_map[data_type]
                if callback_addr in self.callbacks and self.callbacks[callback_addr]:
                    self.callbacks[callback_addr](args)


# ===========================
# MAIN APPLICATION
# ===========================

class LiveConfusionDetector:
    def __init__(self):
        # Core components
        self.detector = RealTimeConfusionDetector()
        self.osc_receiver = OSCReceiver()
        
        # UI components
        self.root = None
        self.text_widget = None
        self.debug_panel = None
        
        # State variables
        self.highlighting_enabled = False
        self.dummy_highlighting_enabled = False
        self.current_paragraph_index = 0
        self.current_word = ""
        self.last_three_words = deque(maxlen=3)
        self.data_flowing = False
        
        # Data storage
        self.latest_eeg = np.zeros(4)
        self.latest_fnirs = np.zeros(8)
        self.latest_motion = np.zeros(6)
        
        # Confusion tracking
        self.word_confusion_map = {}  # word_index -> probability
        self.highlighted_words = set()
        self.manual_highlights = set()  # for dummy mode
        
        # Training texts
        self.training_texts = [
            """To look for goal-related hemodynamic signals in the PPC, we acquired fUS images from NHPs using a miniaturized 15-MHz, linear array transducer placed on the dura via a cranial window. The transducer provided a spatial resolution of 100 μm × 100 μm in-plane, slice thicknesses of ~400 μm, covering a plane with a width of 12.8 mm and penetration depth of 16 mm. We positioned the probe surface-normal in a coronal orientation above the PPC (Figures 1A and 1B). We then selected planes of interest for each animal from the volumes available (Figures 1C–1F). Specifically, we chose planes that captured both the lateral and medial banks of the intraparietal sulcus (ips) within a single image and exhibited behaviorally tuned hemodynamic activity. We used a plane-wave imaging sequence at a pulse repetition frequency of 7,500 Hz and compounded frames collected from a 500-ms period each second to form power Doppler images with a 1 Hz refresh rate.""",

"""To resolve goal-specific hemodynamic changes within single trials, we trained two NHPs to perform memory-delayed instructed saccades. We used a similar task design to previous experiments investigating the roles of PPC regions using fMRI blood-oxygen-level-dependent (BOLD) (Kagan et al., 2010; Wilke et al., 2012) and reversible pharmacological inactivation (Christopoulos et al., 2015). Specifically, the monkeys were required to memorize the location of a cue presented in either the left or right hemifield and execute the movement once the center fixation cue extinguished (Figure 2A). The memory phase was chosen to be sufficiently long (from 4.0 to 5.1 s depending on the animals’ training and success rate, with a mean of 4.4 s across sessions) to capture hemodynamic changes. We collected fUS data while each animal (N = 2) performed memory-delayed saccades. We collected 2,441 trials over 16 days (1,209 from monkey H and 1,232 from monkey L).""",

"""We use statistical parametric maps based on the Student’s t test (one sided with false discovery rate [FDR] correction) to visualize patterns of lateralized activity in PPC (Figures 2B and 2G). We observed event-related average (ERA) changes of cerebral blood volume (CBV) throughout the task from localized regions (Figures 2C–2F, 2H, and 2I). Spatial response fields of laterally tuned hemodynamic activity appeared on the lateral bank of ips (i.e., in LIP). The response fields and ERA waveforms were similar between animals and are consistent with previous electrophysiological (Graf and Andersen, 2014) and fMRI BOLD (Wilke et al., 2012) results. Specifically, ERAs from LIP show higher memory phase responses to contralateral (right)- compared to ipsilateral (left)-cued trials (one-sided t test of area under the curve during memory phase, t test p < 0.001). """,

"""Monkey H exhibited a similar direction-tuned response in the presumed medial parietal area (MP), a small patch of cortex on the medial wall of the hemisphere (we did not record this area effect in monkey L, because MP was outside the imaging plane). This tuning supports previous evidence of MP’s role in directional eye movement observed in a previous study (Thier and Andersen, 1998). In contrast, focal regions of microvasculature outside the LIP also showed strong event-related responses to the task onset but were not tuned to target direction (e.g., Figure 2E). """
        ]
        
        # Thread management
        self.running = False
        
    def setup_ui(self):
        """Create the main UI matching eegtrainer.py visual presentation"""
        self.root = tk.Tk()
        self.root.title("Live Confusion Detection System")
        self.root.geometry("1200x800")  # Match eegtrainer.py

        # Configure dark theme
        self.root.configure(bg='#0a0a0a')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TLabel', background='#1a1a1a', foreground='white')
        style.configure('Dark.TButton', background='#2a2a2a', foreground='white')

        # Main container
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Header (matching eegtrainer.py style)
        header_frame = tk.Frame(main_frame, bg='#1a1a1a', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)

        tk.Label(header_frame,
                text="LIVE CONFUSION DETECTION",
                font=('Arial', 24, 'bold'),
                fg='#FFD93D',
                bg='#1a1a1a').pack(pady=10)

        tk.Label(header_frame,
                text="H: Toggle Highlighting | D: Dummy Mode | Left/Right: Change Text | Space: Toggle Debug",
                font=('Arial', 14),
                fg='#4ECDC4',
                bg='#1a1a1a').pack()

        # Text display frame (matching eegtrainer.py exactly)
        text_frame = tk.Frame(main_frame, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.text_widget = tk.Text(text_frame,
                                   font=('Georgia', 28, 'normal'),  # Match eegtrainer.py
                                   bg='#0a0a0a',
                                   fg='white',
                                   wrap=tk.WORD,
                                   padx=40,  # Match eegtrainer.py
                                   pady=30,  # Match eegtrainer.py
                                   spacing1=10,
                                   spacing2=8,
                                   spacing3=10,
                                   insertwidth=0,
                                   highlightthickness=0,
                                   borderwidth=0,
                                   relief=tk.FLAT,
                                   cursor="hand2")
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Configure text tags
        self.text_widget.tag_configure('confusion_predicted', background='#ff6b6b', foreground='white')
        self.text_widget.tag_configure('manual_highlight', background='#4ecdc4', foreground='black')
        self.text_widget.tag_configure('current_word', underline=True, foreground='#45b7d1')

        # Status frame at bottom (matching eegtrainer.py style)
        status_frame = tk.Frame(main_frame, bg='#1a1a1a', height=120)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)

        # Status labels (compact, like eegtrainer.py)
        self.text_status = tk.Label(status_frame,
                                   text="Text: 1/3",
                                   font=('Arial', 16),
                                   fg='#96CEB4',
                                   bg='#1a1a1a')
        self.text_status.pack(side=tk.LEFT, padx=20, pady=10)

        self.model_status = tk.Label(status_frame,
                                    text="Model: Not Loaded",
                                    font=('Arial', 16),
                                    fg='#888888',
                                    bg='#1a1a1a')
        self.model_status.pack(side=tk.LEFT, padx=20, pady=10)

        self.warmup_status = tk.Label(status_frame,
                                     text="Waiting for data...",
                                     font=('Arial', 16, 'bold'),
                                     fg='#888888',
                                     bg='#1a1a1a')
        self.warmup_status.pack(side=tk.LEFT, padx=20, pady=10)

        self.prediction_status = tk.Label(status_frame,
                                         text="Predictions: 0",
                                         font=('Arial', 16),
                                         fg='#E74C3C',
                                         bg='#1a1a1a')
        self.prediction_status.pack(side=tk.LEFT, padx=20, pady=10)

        self.word_status = tk.Label(status_frame,
                                   text="Current word: -",
                                   font=('Arial', 14, 'italic'),
                                   fg='#FFD93D',
                                   bg='#1a1a1a')
        self.word_status.pack(side=tk.RIGHT, padx=20, pady=5)

        # Debug panel (togglable, hidden by default)
        self.debug_panel_visible = False
        self.debug_window = None

        # Load initial text
        self.load_text(0)

        # Bind events
        self.text_widget.bind('<Motion>', self.on_mouse_move)
        self.text_widget.bind('<Button-1>', self.on_left_click)
        self.text_widget.bind('<Button-3>', self.on_right_click)
        self.root.bind('<Key>', self.on_key_press)

        # Start cursor tracking
        self.root.after(50, self.track_cursor)
    
    
    def update_debug_info(self):
        """Update status bar information"""
        # Update text status
        if hasattr(self, 'text_status'):
            self.text_status.config(text=f"Text: {self.current_paragraph_index + 1}/{len(self.training_texts)}")

        # Update model status
        if hasattr(self, 'model_status'):
            if self.detector.model_loaded:
                self.model_status.config(text="Model: Loaded", fg='#96CEB4')
            else:
                self.model_status.config(text="Model: Not Loaded", fg='#888888')

        # Update warmup status
        if hasattr(self, 'warmup_status'):
            if self.detector.buffer_ready:
                if self.data_flowing:
                    self.warmup_status.config(text="READY - Predicting", fg='#96CEB4')
                else:
                    self.warmup_status.config(text="Buffer Ready", fg='#FFD93D')
            elif self.detector.first_data_time:
                elapsed = time.time() - self.detector.first_data_time
                remaining = max(0, self.detector.warmup_duration - elapsed)
                self.warmup_status.config(text=f"Warming up... {remaining:.1f}s", fg='#FFD93D')
            else:
                self.warmup_status.config(text="Waiting for data...", fg='#888888')

        # Update prediction count
        if hasattr(self, 'prediction_status'):
            n_predictions = len(self.detector.confused_word_positions)
            self.prediction_status.config(text=f"Confused Words: {n_predictions}")

        # Update current word
        if hasattr(self, 'word_status'):
            self.word_status.config(text=f"Current word: {self.current_word if self.current_word else '-'}")
    
    def load_text(self, index):
        """Load a training text"""
        self.current_paragraph_index = index
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, self.training_texts[index])

        # Reset confusion tracking
        self.detector.confused_word_positions.clear()
        self.word_confusion_map.clear()
        self.highlighted_words.clear()
        self.manual_highlights.clear()

        # Clear prediction history
        self.detector.word_predictions.clear()
        self.detector.word_reading_history.clear()
    
    def track_cursor(self):
        """Track cursor position and current word"""
        if not self.running:
            return

        try:
            # Process delayed predictions first
            if self.detector.model_loaded and self.data_flowing:
                self.detector.process_delayed_predictions()

            # Get cursor position
            cursor_pos = self.text_widget.index(tk.CURRENT)

            # Get word at cursor
            word_start = self.text_widget.index(f"{cursor_pos} wordstart")
            word_end = self.text_widget.index(f"{cursor_pos} wordend")
            word = self.text_widget.get(word_start, word_end).strip()

            if word and word != self.current_word:
                # Update current word
                self.current_word = word
                self.last_three_words.append(word)

                # Clear previous current word highlighting
                self.text_widget.tag_remove('current_word', 1.0, tk.END)
                self.text_widget.tag_add('current_word', word_start, word_end)

                # Store word for delayed prediction with its text position
                if self.detector.model_loaded and self.data_flowing:
                    self.detector.update_word_reading_history(word, word_start)

            # Update highlighting based on smoothed predictions
            if self.detector.model_loaded and self.highlighting_enabled:
                self.update_confusion_highlighting()

            # Update debug panel
            self.update_debug_info()

        except Exception as e:
            print(f"Cursor tracking error: {e}")

        # Schedule next update
        self.root.after(50, self.track_cursor)

    def update_confusion_highlighting(self):
        """Update text highlighting based on confirmed confused words at specific positions"""
        current_time = time.time()

        # Check for NEW confusing words and add their positions to the persistent set
        for (word, timestamp), pred_data in self.detector.word_predictions.items():
            smoothed_prob = self.detector.get_smoothed_confusion_probability(word, current_time)

            if smoothed_prob > self.detector.confusion_threshold:
                # Add this specific word position (not all instances of the word)
                text_position = pred_data.get('text_position')
                if text_position:
                    self.detector.confused_word_positions.add((word, text_position))

        # Now highlight ONLY the specific positions that were predicted as confusing
        self.text_widget.tag_remove('confusion_predicted', 1.0, tk.END)
        self.highlighted_words.clear()

        # Validate and clean up confused_word_positions
        valid_positions = set()

        for word, position in self.detector.confused_word_positions:
            try:
                # Verify position is still valid in current text
                word_start = self.text_widget.index(f"{position} wordstart")
                word_end = self.text_widget.index(f"{position} wordend")

                # Verify the word at this position matches what was predicted
                actual_word = self.text_widget.get(word_start, word_end).strip()
                if actual_word.lower() == word.lower():
                    self.text_widget.tag_add('confusion_predicted', word_start, word_end)
                    self.highlighted_words.add(position)
                    valid_positions.add((word, position))
                # If word doesn't match, position is stale - don't add to valid_positions
            except tk.TclError:
                # Position no longer valid (text changed) - don't add to valid_positions
                pass

        # Update confused_word_positions to only keep valid positions
        self.detector.confused_word_positions = valid_positions
    
    def on_mouse_move(self, event):
        """Handle mouse movement"""
        # Update cursor position for word tracking
        self.text_widget.mark_set(tk.CURRENT, f"@{event.x},{event.y}")
    
    def on_left_click(self, event):
        """Handle left click - word selection in dummy mode"""
        if self.dummy_highlighting_enabled:
            cursor_pos = self.text_widget.index(f"@{event.x},{event.y}")
            word_start = self.text_widget.index(f"{cursor_pos} wordstart")
            word_end = self.text_widget.index(f"{cursor_pos} wordend")
            
            word_index = self.text_widget.index(word_start)
            if word_index in self.manual_highlights:
                self.text_widget.tag_remove('manual_highlight', word_start, word_end)
                self.manual_highlights.remove(word_index)
            else:
                self.text_widget.tag_add('manual_highlight', word_start, word_end)
                self.manual_highlights.add(word_index)
    
    def on_right_click(self, event):
        """Handle right click - sentence selection in dummy mode"""
        if self.dummy_highlighting_enabled:
            cursor_pos = self.text_widget.index(f"@{event.x},{event.y}")
            
            # Find sentence boundaries
            text = self.text_widget.get(1.0, tk.END)
            index = int(cursor_pos.split('.')[1])
            
            # Simple sentence detection
            sentence_start = text.rfind('.', 0, index) + 1
            if sentence_start == 0:
                sentence_start = 0
            else:
                sentence_start += 1
                
            sentence_end = text.find('.', index)
            if sentence_end == -1:
                sentence_end = len(text)
            else:
                sentence_end += 1
            
            # Convert to text indices
            start_idx = f"1.{sentence_start}"
            end_idx = f"1.{sentence_end}"
            
            # Toggle highlight
            if self.text_widget.tag_ranges('manual_highlight'):
                # Check if sentence already highlighted
                existing_tags = self.text_widget.tag_names(start_idx)
                if 'manual_highlight' in existing_tags:
                    self.text_widget.tag_remove('manual_highlight', start_idx, end_idx)
                else:
                    self.text_widget.tag_add('manual_highlight', start_idx, end_idx)
            else:
                self.text_widget.tag_add('manual_highlight', start_idx, end_idx)
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.char.lower() == 'h':
            self.toggle_highlighting()
        elif event.char.lower() == 'd':
            self.toggle_dummy_mode()
        elif event.char == ' ':
            self.toggle_debug_window()
        elif event.keysym == 'Left':
            new_idx = (self.current_paragraph_index - 1) % len(self.training_texts)
            self.load_text(new_idx)
        elif event.keysym == 'Right':
            new_idx = (self.current_paragraph_index + 1) % len(self.training_texts)
            self.load_text(new_idx)
        elif event.char.lower() == 'q':
            self.quit()
    
    def toggle_highlighting(self):
        """Toggle confusion highlighting"""
        self.highlighting_enabled = not self.highlighting_enabled

        if self.highlighting_enabled:
            # Trigger immediate update
            self.update_confusion_highlighting()
            print("Highlighting enabled - predictions will appear with 1.5s delay")
        else:
            # Remove all confusion highlights
            self.text_widget.tag_remove('confusion_predicted', 1.0, tk.END)
            self.highlighted_words.clear()
            print("Highlighting disabled")
    
    def toggle_dummy_mode(self):
        """Toggle dummy highlighting mode"""
        self.dummy_highlighting_enabled = not self.dummy_highlighting_enabled

        if self.dummy_highlighting_enabled:
            print("Dummy highlighting mode enabled - Click words or right-click sentences")
        else:
            # Clear manual highlights
            self.text_widget.tag_remove('manual_highlight', 1.0, tk.END)
            self.manual_highlights.clear()
            print("Dummy highlighting mode disabled")

    def toggle_debug_window(self):
        """Toggle debug information window"""
        if self.debug_window and self.debug_window.winfo_exists():
            # Close existing debug window
            self.debug_window.destroy()
            self.debug_window = None
            self.debug_panel_visible = False
            print("Debug panel closed")
        else:
            # Create debug window
            self.debug_window = tk.Toplevel(self.root)
            self.debug_window.title("Debug Information")
            self.debug_window.geometry("500x700")
            self.debug_window.configure(bg='#1a1a1a')

            # Create debug info display
            debug_frame = tk.Frame(self.debug_window, bg='#1a1a1a')
            debug_frame.pack(fill='both', expand=True, padx=20, pady=20)

            title = tk.Label(debug_frame, text="DEBUG PANEL",
                           font=('Arial', 18, 'bold'), fg='#FFD93D', bg='#1a1a1a')
            title.pack(pady=(0, 20))

            # Create scrollable text widget for debug info
            self.debug_text = scrolledtext.ScrolledText(debug_frame,
                                                        font=('Consolas', 10),
                                                        bg='#0a0a0a',
                                                        fg='#96CEB4',
                                                        wrap=tk.WORD,
                                                        height=35)
            self.debug_text.pack(fill='both', expand=True)

            # Load model button
            load_button = tk.Button(debug_frame, text="Load Model",
                                   command=self.load_model_dialog,
                                   bg='#2a2a2a', fg='white',
                                   font=('Arial', 12))
            load_button.pack(pady=10)

            self.debug_panel_visible = True
            print("Debug panel opened")

            # Start updating debug window
            self.update_debug_window()

    def update_debug_window(self):
        """Update debug window content"""
        if not self.debug_panel_visible or not self.debug_window or not self.debug_window.winfo_exists():
            return

        try:
            # Build debug info text
            debug_info = []
            debug_info.append("=" * 50)
            debug_info.append("LIVE CONFUSION DETECTION - DEBUG PANEL")
            debug_info.append("=" * 50)
            debug_info.append("")

            # 1. Current word
            debug_info.append("1. CURRENT WORD:")
            debug_info.append(f"   {self.current_word if self.current_word else 'None'}")
            debug_info.append("")

            # 2. Signal values
            debug_info.append("2. SIGNAL VALUES:")
            debug_info.append(f"   EEG: [{', '.join(f'{x:.1f}' for x in self.latest_eeg)}]")
            debug_info.append(f"   fNIRS: [{', '.join(f'{x:.1f}' for x in self.latest_fnirs)}]")
            debug_info.append(f"   Motion: [{', '.join(f'{x:.1f}' for x in self.latest_motion)}]")
            debug_info.append("")

            # 3. Last three words
            debug_info.append("3. LAST THREE WORDS:")
            if self.last_three_words:
                current_time = time.time()
                for word in self.last_three_words:
                    complexity = self.detector.complexity_analyzer.analyze_complexity(word)
                    prob = self.detector.get_smoothed_confusion_probability(word, current_time)
                    debug_info.append(f"   {word}: grade={complexity['estimated_grade_level']:.1f}, conf={prob:.2f}")
            else:
                debug_info.append("   None")
            debug_info.append("")

            # 4. Model status
            debug_info.append("4. MODEL STATUS:")
            if self.detector.model_loaded:
                n_predictions = len(self.detector.word_predictions)
                n_candidates = len(self.detector.get_delayed_prediction_candidates())
                debug_info.append(f"   Loaded: Yes")
                debug_info.append(f"   Predictions made: {n_predictions}")
                debug_info.append(f"   Pending predictions: {n_candidates}")
            else:
                debug_info.append("   Loaded: No")
            debug_info.append("")

            # 5. Data flow
            debug_info.append("5. DATA FLOW:")
            debug_info.append(f"   Status: {'Flowing' if self.data_flowing else 'Not flowing'}")
            if self.data_flowing and self.detector.model_loaded:
                buffer_fill = len(self.detector.eeg_buffer) / self.detector.window_size
                debug_info.append(f"   Buffer fill: {buffer_fill:.0%}")
            debug_info.append("")

            # 6. Buffer status
            debug_info.append("6. BUFFER STATUS:")
            debug_info.append(f"   Ready: {self.detector.buffer_ready}")
            if self.detector.first_data_time:
                elapsed = time.time() - self.detector.first_data_time
                debug_info.append(f"   Data collection time: {elapsed:.1f}s")
            debug_info.append(f"   EEG buffer size: {len(self.detector.eeg_buffer)}/{self.detector.window_size}")
            debug_info.append("")

            # 7. Confused words
            debug_info.append("7. PREDICTED CONFUSED WORDS:")
            if self.detector.confused_word_positions:
                current_time = time.time()
                word_probs = []
                unique_words = set(word for word, pos in self.detector.confused_word_positions)
                for word in unique_words:
                    prob = self.detector.get_smoothed_confusion_probability(word, current_time)
                    word_probs.append((word, prob))

                word_probs.sort(key=lambda x: x[1], reverse=True)

                for word, prob in word_probs[:10]:
                    debug_info.append(f"   {word}: {prob:.3f}")

                if len(unique_words) > 10:
                    debug_info.append(f"   ... and {len(unique_words)-10} more")
            else:
                debug_info.append("   None")
            debug_info.append("")

            # 8. Reading state
            debug_info.append("8. READING STATE:")
            debug_info.append(f"   Active: {self.detector.reading_active}")
            debug_info.append(f"   Time since last word: {time.time() - self.detector.last_word_change_time:.1f}s")
            debug_info.append("")

            # Update debug text widget
            self.debug_text.delete(1.0, tk.END)
            self.debug_text.insert(1.0, "\n".join(debug_info))

            # Schedule next update
            if self.debug_panel_visible:
                self.root.after(100, self.update_debug_window)

        except Exception as e:
            print(f"Debug window update error: {e}")

    def load_model_dialog(self):
        """Open file dialog to load model"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        
        if filename:
            self.detector.load_model(filename)
            self.update_debug_info()
    
    def on_eeg_data(self, values):
        """Handle EEG data callback"""
        if len(values) == 4:
            self.latest_eeg = np.array(values)
            self.detector.add_eeg_sample(values)
            self.data_flowing = True

    def on_fnirs_data(self, values):
        """Handle fNIRS data callback"""
        if len(values) == 8:
            self.latest_fnirs = np.array(values)
            self.detector.add_fnirs_sample(values)

    def on_acc_data(self, values):
        """Handle accelerometer data callback"""
        if len(values) == 3:
            self.latest_motion[:3] = values
            # Motion data needs to be combined (acc + gyro = 6 values)
            self.detector.add_motion_sample(self.latest_motion)

    def on_gyro_data(self, values):
        """Handle gyroscope data callback"""
        if len(values) == 3:
            self.latest_motion[3:] = values
            # Motion data needs to be combined (acc + gyro = 6 values)
            self.detector.add_motion_sample(self.latest_motion)
    
    
    def run(self):
        """Start the application"""
        print("=" * 60)
        print("   LIVE CONFUSION DETECTION SYSTEM")
        print("=" * 60)
        print("\n📡 Starting OSC receiver on port 8052...")
        
        # Setup OSC callbacks
        self.osc_receiver.set_callback('/muse/eeg', self.on_eeg_data)
        self.osc_receiver.set_callback('/muse/optics', self.on_fnirs_data)
        self.osc_receiver.set_callback('/muse/acc', self.on_acc_data)
        self.osc_receiver.set_callback('/muse/gyro', self.on_gyro_data)
        
        # Start OSC receiver
        self.osc_receiver.start()
        
        # Start main loop
        self.running = True
        
        print("\n🖥️ Starting UI...")
        print("\nKEYBOARD SHORTCUTS:")
        print("  H - Toggle confusion highlighting")
        print("  D - Toggle dummy highlighting mode")
        print("  ←/→ - Change text passage")
        print("  Q - Quit application")
        print("\nIMPORTANT:")
        print("  System requires 3.5s buffer warmup before making predictions.")
        print("  After warmup, predictions use 1.5s delay to match training temporal alignment.")
        print("  Words will be highlighted ~1.5 seconds after you read them.")
        print("  Confusion threshold set to 70% to reduce false positives.")
        print("\n✓ System ready!")
        
        # Setup and run UI
        self.setup_ui()
        self.root.mainloop()
    
    def quit(self):
        """Clean shutdown"""
        print("\nShutting down...")
        self.running = False
        self.osc_receiver.stop()
        if self.root:
            self.root.quit()


# ===========================
# MAIN ENTRY POINT
# ===========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Confusion Detection System")
    parser.add_argument('--model', type=str, help='Path to pre-trained model file')
    parser.add_argument('--port', type=int, default=8052, help='OSC UDP port (default: 8052)')
    args = parser.parse_args()
    
    # Create and run application
    app = LiveConfusionDetector()
    if args.model:
        app.detector.load_model(args.model)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()