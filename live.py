#!/usr/bin/env python3
"""
live.py - Real-time Confusion Detection System
Combines EEG/fNIRS data collection with neural network inference
to predict reading confusion in real-time.
"""

//github

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

class RealTimeConfusionDetector:
    def __init__(self, model_path=None):
        # Neural network components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False

        # Data buffers - using 512 samples to match model training
        self.window_size = 512  # 2 seconds at 256 Hz (matches neural.py)
        self.eeg_buffer = deque(maxlen=self.window_size)
        self.fnirs_buffer = deque(maxlen=self.window_size)
        self.motion_buffer = deque(maxlen=self.window_size)

        # Word tracking with timing (matching neural.py's pre-event offset)
        self.current_word = ""
        self.word_history = deque(maxlen=50)  # Store more history for delayed prediction
        self.word_reading_history = deque(maxlen=50)  # (word, timestamp, buffer_state)
        self.word_predictions = {}
        self.complexity_analyzer = WordComplexityAnalyzer()

        # Timing parameters (matching neural.py)
        self.pre_event_offset = 2.0  # Match neural.py training offset

        # Reading state detection
        self.last_word_change_time = time.time()
        self.reading_active = False
        self.cursor_stationary_threshold = 1.0  # seconds
        self.words_per_minute_threshold = 50  # minimum reading speed

        # Confusion tracking
        self.paragraph_confusion_words = set()
        self.confusion_threshold = 0.3  # Lower threshold for real-time (was 0.5)

        # Initialize with zeros
        for _ in range(self.window_size):
            self.eeg_buffer.append(np.zeros(4))
            self.fnirs_buffer.append(np.zeros(8))
            self.motion_buffer.append(np.zeros(6))

        # Feature extraction parameters
        self.sample_rate = 256
        self.feature_names = None

        # Thread safety
        self.lock = threading.Lock()

        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load pre-trained model and associated components"""
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
            
            # Load window parameters
            self.window_size = int(checkpoint.get('window_size', 3.0) * checkpoint.get('sample_rate', 256))
            self.sample_rate = checkpoint.get('sample_rate', 256)
            
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
            print(f"  Model expects {model_config.get('n_features', 100)} features")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def add_eeg_sample(self, eeg_data):
        """Add EEG sample to buffer thread-safely"""
        if len(eeg_data) == 4:
            with self.lock:
                self.eeg_buffer.append(np.array(eeg_data))

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

    def update_word_reading_history(self, word):
        """Update word reading history with current buffer state"""
        if not word or word == self.current_word:
            return

        current_time = time.time()
        self.last_word_change_time = current_time

        # Store word with timestamp and current buffer state
        with self.lock:
            # Capture current buffer state for delayed prediction
            eeg_snapshot = np.vstack(list(self.eeg_buffer)).copy() if len(self.eeg_buffer) == self.window_size else None
            fnirs_snapshot = np.vstack(list(self.fnirs_buffer)).copy() if len(self.fnirs_buffer) == self.window_size else None
            motion_snapshot = np.vstack(list(self.motion_buffer)).copy() if len(self.motion_buffer) == self.window_size else None

            if eeg_snapshot is not None and fnirs_snapshot is not None and motion_snapshot is not None:
                self.word_reading_history.append({
                    'word': word,
                    'timestamp': current_time,
                    'eeg': eeg_snapshot,
                    'fnirs': fnirs_snapshot,
                    'motion': motion_snapshot
                })

        self.current_word = word
        self.word_history.append(word)

        # Update reading state
        self.detect_reading_state()

    def get_delayed_prediction_candidates(self):
        """Get words that should be predicted based on pre-event offset"""
        current_time = time.time()
        candidates = []

        for entry in self.word_reading_history:
            time_diff = current_time - entry['timestamp']
            # Check if this word was read approximately pre_event_offset seconds ago
            if abs(time_diff - self.pre_event_offset) < 0.5:  # Within 0.5 seconds of target delay
                candidates.append(entry)

        return candidates
    
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
            """Epineural cuff electrodes are designed to interface with peripheral nerves by wrapping around the epineurium. These devices utilize biocompatible materials like silicone or polyimide to create a cylindrical structure that gently encompasses the nerve bundle. The electrode contacts, typically made of platinum-iridium or gold, are embedded within the cuff material and positioned to make electrical contact with the nerve fibers through the epineurium. This configuration allows for selective stimulation and recording of neural signals while minimizing invasive penetration of the nerve tissue. The cuff design provides mechanical stability and prevents electrode migration, which is crucial for chronic implantation scenarios in neuroprosthetic applications.""",
            
            """The biomechanical properties of neural interfaces significantly impact their long-term functionality and biocompatibility. When designing epineural electrodes, engineers must consider the mechanical mismatch between rigid electronic materials and soft neural tissue. This disparity can lead to chronic inflammation, scar tissue formation, and eventual signal degradation. Advanced fabrication techniques now incorporate flexible substrates and stretchable conductors that better match the Young's modulus of nerve tissue. Additionally, the incorporation of anti-inflammatory coatings and drug-eluting polymers has shown promise in reducing the foreign body response. These innovations aim to create a more stable biological-electrical interface that maintains signal quality over extended implantation periods."""
        ]
        
        # Thread management
        self.running = False
        
    def setup_ui(self):
        """Create the main UI with text display and debug panel"""
        self.root = tk.Tk()
        self.root.title("Live Confusion Detection System")
        self.root.geometry("1400x900")
        
        # Configure dark theme
        self.root.configure(bg='#0a0a0a')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TLabel', background='#1a1a1a', foreground='white')
        style.configure('Dark.TButton', background='#2a2a2a', foreground='white')
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Text display
        left_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(left_frame, text="LIVE CONFUSION DETECTION", 
                               font=('Arial', 24, 'bold'), style='Dark.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(left_frame, 
            text="Read the text below. The system will predict confusing words in real-time.\n" +
                 "Press 'H' to toggle highlighting | 'D' for dummy mode | ←/→ to change text",
            style='Dark.TLabel', font=('Arial', 12))
        instructions.pack(pady=(0, 10))
        
        # Text display
        text_frame = tk.Frame(left_frame, bg='#0a0a0a', highlightthickness=2, 
                             highlightbackground='#3a3a3a')
        text_frame.pack(fill='both', expand=True)
        
        self.text_widget = tk.Text(text_frame, wrap='word', font=('Georgia', 16),
                                  bg='#0a0a0a', fg='white', insertbackground='white',
                                  selectbackground='#4444ff', selectforeground='white',
                                  padx=20, pady=20, spacing1=10, spacing2=8, spacing3=10)
        self.text_widget.pack(fill='both', expand=True)
        
        # Configure text tags
        self.text_widget.tag_configure('confusion_predicted', background='#ff6b6b', foreground='black')
        self.text_widget.tag_configure('manual_highlight', background='#4ecdc4', foreground='black')
        self.text_widget.tag_configure('current_word', underline=True, foreground='#45b7d1')
        
        # Right side - Debug panel
        self.create_debug_panel(main_frame)
        
        # Load initial text
        self.load_text(0)
        
        # Bind events
        self.text_widget.bind('<Motion>', self.on_mouse_move)
        self.text_widget.bind('<Button-1>', self.on_left_click)
        self.text_widget.bind('<Button-3>', self.on_right_click)
        self.root.bind('<Key>', self.on_key_press)
        
        # Start cursor tracking
        self.root.after(50, self.track_cursor)
    
    def create_debug_panel(self, parent):
        """Create the debug information panel"""
        debug_frame = ttk.Frame(parent, style='Dark.TFrame', width=400)
        debug_frame.pack(side='right', fill='y', padx=(10, 0))
        debug_frame.pack_propagate(False)
        
        # Title
        title = ttk.Label(debug_frame, text="DEBUG PANEL", 
                         font=('Arial', 18, 'bold'), style='Dark.TLabel')
        title.pack(pady=(0, 20))
        
        # Debug info sections
        self.debug_info = {}
        
        # 1. Current word
        self.add_debug_section(debug_frame, "1. Current Word", "current_word", "None")
        
        # 2. Signal values
        self.add_debug_section(debug_frame, "2. Signal Values", "signals", 
                              "EEG: [0, 0, 0, 0]\nfNIRS: [0, 0, 0, 0, 0, 0, 0, 0]\nMotion: [0, 0, 0, 0, 0, 0]")
        
        # 3. Last three words
        self.add_debug_section(debug_frame, "3. Last Three Words", "word_history", "None")
        
        # 4. Model status
        self.add_debug_section(debug_frame, "4. Neural Net Status", "model_status", "Not loaded")
        
        # 5. Data flow status
        self.add_debug_section(debug_frame, "5. Data Flow", "data_flow", "Not connected")
        
        # 6. Confused words
        self.add_debug_section(debug_frame, "6. Predicted Confusions", "confused_words", "None")
        
        # 7. Dummy highlighting button
        dummy_frame = ttk.Frame(debug_frame, style='Dark.TFrame')
        dummy_frame.pack(fill='x', pady=10)
        
        dummy_label = ttk.Label(dummy_frame, text="7. Test Highlighting", 
                               font=('Arial', 12, 'bold'), style='Dark.TLabel')
        dummy_label.pack(anchor='w')
        
        self.dummy_button = ttk.Button(dummy_frame, text="Enable Dummy Mode",
                                      command=self.toggle_dummy_mode,
                                      style='Dark.TButton')
        self.dummy_button.pack(pady=5)
        
        # Load model button
        load_button = ttk.Button(debug_frame, text="Load Model",
                               command=self.load_model_dialog,
                               style='Dark.TButton')
        load_button.pack(pady=10)
        
    def add_debug_section(self, parent, title, key, initial_value):
        """Add a debug information section"""
        frame = ttk.Frame(parent, style='Dark.TFrame')
        frame.pack(fill='x', pady=10)
        
        label = ttk.Label(frame, text=title, font=('Arial', 12, 'bold'), 
                         style='Dark.TLabel')
        label.pack(anchor='w')
        
        value_label = ttk.Label(frame, text=initial_value, style='Dark.TLabel',
                               font=('Consolas', 10), wraplength=350)
        value_label.pack(anchor='w', padx=(10, 0))
        
        self.debug_info[key] = value_label
    
    def update_debug_info(self):
        """Update debug panel information"""
        # 1. Current word
        self.debug_info['current_word'].config(text=self.current_word or "None")
        
        # 2. Signal values
        signal_text = f"EEG: [{', '.join(f'{x:.1f}' for x in self.latest_eeg)}]\n"
        signal_text += f"fNIRS: [{', '.join(f'{x:.1f}' for x in self.latest_fnirs[:4])}...]\n"
        signal_text += f"Motion: [{', '.join(f'{x:.1f}' for x in self.latest_motion[:3])}...]"
        self.debug_info['signals'].config(text=signal_text)
        
        # 3. Last three words with complexity
        if self.last_three_words:
            word_text = ""
            for word in self.last_three_words:
                complexity = self.detector.complexity_analyzer.analyze_complexity(word)
                word_text += f"{word}: grade={complexity['estimated_grade_level']:.1f}\n"
            self.debug_info['word_history'].config(text=word_text.strip())
        
        # 4. Model status
        status = "✓ Loaded" if self.detector.model_loaded else "✗ Not loaded"
        self.debug_info['model_status'].config(text=status)
        
        # 5. Data flow
        flow_status = "✓ Flowing" if self.data_flowing else "✗ Not flowing"
        self.debug_info['data_flow'].config(text=flow_status)
        
        # 6. Confused words
        if self.detector.paragraph_confusion_words:
            confused_text = ", ".join(sorted(self.detector.paragraph_confusion_words)[:5])
            if len(self.detector.paragraph_confusion_words) > 5:
                confused_text += f" (+{len(self.detector.paragraph_confusion_words)-5} more)"
        else:
            confused_text = "None"
        self.debug_info['confused_words'].config(text=confused_text)
    
    def load_text(self, index):
        """Load a training text"""
        self.current_paragraph_index = index
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, self.training_texts[index])
        
        # Reset confusion tracking
        self.detector.paragraph_confusion_words.clear()
        self.word_confusion_map.clear()
        self.highlighted_words.clear()
        self.manual_highlights.clear()
    
    def track_cursor(self):
        """Track cursor position and current word"""
        if not self.running:
            return
            
        try:
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
                
                # Predict confusion for this word
                if self.detector.model_loaded and self.data_flowing:
                    prob, pred_class = self.detector.predict_confusion(word)
                    
                    # Store prediction
                    word_index = self.text_widget.index(word_start)
                    self.word_confusion_map[word_index] = prob
                    
                    # Update confusion set if above threshold
                    if prob > self.detector.confusion_threshold:
                        self.detector.paragraph_confusion_words.add(word)
                        
                        # Highlight if enabled
                        if self.highlighting_enabled and word_index not in self.highlighted_words:
                            self.text_widget.tag_add('confusion_predicted', word_start, word_end)
                            self.highlighted_words.add(word_index)
            
            # Update debug panel
            self.update_debug_info()
            
        except Exception as e:
            print(f"Cursor tracking error: {e}")
        
        # Schedule next update
        self.root.after(50, self.track_cursor)
    
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
            # Show all predicted confusions
            for word_idx, prob in self.word_confusion_map.items():
                if prob > self.detector.confusion_threshold:
                    word_start = word_idx
                    word_end = self.text_widget.index(f"{word_idx} wordend")
                    self.text_widget.tag_add('confusion_predicted', word_start, word_end)
            print("✓ Highlighting enabled")
        else:
            # Remove all confusion highlights
            self.text_widget.tag_remove('confusion_predicted', 1.0, tk.END)
            print("✗ Highlighting disabled")
    
    def toggle_dummy_mode(self):
        """Toggle dummy highlighting mode"""
        self.dummy_highlighting_enabled = not self.dummy_highlighting_enabled
        
        if self.dummy_highlighting_enabled:
            self.dummy_button.config(text="Disable Dummy Mode")
            self.text_widget.config(cursor="hand2")
            print("✓ Dummy highlighting mode enabled - Click words or right-click sentences")
        else:
            self.dummy_button.config(text="Enable Dummy Mode")
            self.text_widget.config(cursor="xterm")
            # Clear manual highlights
            self.text_widget.tag_remove('manual_highlight', 1.0, tk.END)
            self.manual_highlights.clear()
            print("✗ Dummy highlighting mode disabled")
    
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