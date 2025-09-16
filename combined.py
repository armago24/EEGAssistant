#!/usr/bin/env python3
"""
Real-time Word Confusion Detector
Combines EEG/fNIRS data collection with ML-based confusion detection
Highlights confusing words/sentences in real-time while reading
"""

import socket
import struct
import threading
import time
import numpy as np
from collections import deque, defaultdict
from scipy import signal, stats
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import os
from datetime import datetime
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import font as tkfont
import signal as sig
import atexit
import sys
import glob

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Training texts (same as before)
TRAINING_TEXTS = [
"""2.3. Extracellular electrical stimulation — The extracellular tissue medium was approximated as an isotropic ohmic conductor. A disk electrode acting as a current source with ground located at infinity was placed below (∼10 µm) the modeled RGC. Since the extracellular domain was isotropic, the extracellular voltage at each point in space was calculated in a similar fashion to previous work [16] (equation (5)). R is the radius of the disk electrode (5 µm), I is the stimulation current, and ρe is the extracellular resistivity (1000 Ω·cm). A large range of resistivity values for the ganglion cell layer is reported (500–7900 Ω·cm). Here, 1000 Ω·cm, used in several studies, best matched the experimental data. The electrode array was placed 10 µm below the nerve-fiber layer.""",

"""Electrical stimuli consisted of triphasic charge-balanced pulses with relative current amplitudes 2:−3:1 and duration of 50 µs per phase (150 µs total), matching experimental stimulation parameters. The activation threshold was defined as the lowest current amplitude required to elicit a spike. For multi-electrode stimulation, for each fixed ratio of current levels through the electrodes, the smallest current amplitude combination that produced a spike was identified as the activation threshold.""",
]

class RealTimeConfusionDetector:
    def __init__(self, port=8052, buffer_size=2000, window_duration=10):
        # Network settings
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = buffer_size
        self.window_duration = window_duration
        self.timestamps = deque(maxlen=buffer_size)
        
        # Channel data storage
        self.eeg_channels = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size),
            'AF8': deque(maxlen=buffer_size),
            'TP10': deque(maxlen=buffer_size)
        }
        
        self.fnirs_channels = {
            'Ch1_norm': deque(maxlen=buffer_size),
            'Ch2_norm': deque(maxlen=buffer_size),
            'Ch3_norm': deque(maxlen=buffer_size),
            'Ch4_norm': deque(maxlen=buffer_size),
            'Ch1_raw': deque(maxlen=buffer_size),
            'Ch2_raw': deque(maxlen=buffer_size),
            'Ch3_raw': deque(maxlen=buffer_size),
            'Ch4_raw': deque(maxlen=buffer_size)
        }
        
        self.motion_channels = {
            'acc_x': deque(maxlen=buffer_size),
            'acc_y': deque(maxlen=buffer_size),
            'acc_z': deque(maxlen=buffer_size),
            'gyro_x': deque(maxlen=buffer_size),
            'gyro_y': deque(maxlen=buffer_size),
            'gyro_z': deque(maxlen=buffer_size)
        }
        
        self.ref_channels = {
            'DRL': deque(maxlen=buffer_size),
            'REF': deque(maxlen=buffer_size)
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # ML Model components
        self.classifier = None
        self.scaler = None
        self.feature_names = []
        self.sample_rate = 256
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
        # Real-time prediction
        self.prediction_window = 2.0  # seconds
        self.prediction_threshold = 0.5  # probability threshold for confusion
        self.word_predictions = {}  # word -> confusion probability
        self.prediction_update_interval = 0.1  # seconds
        self.last_prediction_time = 0
        
        # Preprocessing buffers
        self.eeg_processed = {}
        for ch in self.channels:
            self.eeg_processed[ch] = deque(maxlen=buffer_size)
        
        # Current state
        self.current_word = ""
        self.current_text_index = 0
        
        # Visualization
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.teleprompter = None
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # For tracking the last saved data
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Confusion tracking for validation
        self.manual_confusion_words = set()  # Words manually marked as confusing
        
        # Colors
        self.eeg_colors = {
            'TP9': '#FF6B6B',
            'AF7': '#4ECDC4',
            'AF8': '#45B7D1',
            'TP10': '#96CEB4'
        }
        
        self.fnirs_colors = {
            'Ch1': '#E74C3C',
            'Ch2': '#3498DB',
            'Ch3': '#2ECC71',
            'Ch4': '#F39C12'
        }
        
        # Flag to track if we're in the process of shutting down
        self.shutting_down = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup_on_exit)
        sig.signal(sig.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nReceived interrupt signal. Shutting down...")
        self.cleanup_on_exit()
        sys.exit(0)
    
    def cleanup_on_exit(self):
        """Cleanup function called on exit"""
        if self.shutting_down:
            return
        self.shutting_down = True
        
        self.running = False
        if self.socket:
            self.socket.close()
    
    def load_training_data(self, filepaths):
        """Load training data and train classifier"""
        print(f"\n{'='*50}")
        print("LOADING TRAINING DATA")
        print(f"{'='*50}")
        
        all_X = []
        all_y = []
        
        for filepath in filepaths:
            print(f"\nLoading: {filepath}")
            data = np.load(filepath, allow_pickle=True)
            
            # Extract arrays
            timestamps = data['timestamps']
            eeg = data['eeg']
            fnirs = data['fnirs'] if 'fnirs' in data else None
            event_timestamps = data['event_timestamps']
            event_types = data['event_types']
            event_words = data['event_words'] if 'event_words' in data else None
            tracked_words = data['words'] if 'words' in data else None
            
            # Process events
            confused_words = defaultdict(list)
            if event_words is not None:
                for timestamp, event_type, word in zip(event_timestamps, event_types, event_words):
                    if word and isinstance(word, str) and word.strip():
                        clean_word = word.strip().lower()
                        if event_type == 'word_confusion':
                            confused_words[clean_word].append((timestamp, 'word'))
                        elif event_type == 'sentence_confusion':
                            confused_words[clean_word].append((timestamp, 'sentence'))
            
            # Build timeline
            all_words_timeline = []
            if tracked_words is not None:
                for timestamp, word in zip(timestamps, tracked_words):
                    if word and isinstance(word, str) and word.strip():
                        clean_word = word.strip().lower()
                        all_words_timeline.append((timestamp, clean_word))
            
            # Preprocess signals
            eeg_processed = self._preprocess_eeg_batch(eeg)
            fnirs_processed = self._preprocess_fnirs_batch(fnirs) if fnirs is not None else None
            
            # Extract features for confused words
            for word, events in confused_words.items():
                for timestamp, conf_type in events:
                    features = self._extract_features_batch(
                        timestamp, word, timestamps, eeg_processed, fnirs_processed
                    )
                    if features is not None:
                        all_X.append(features)
                        all_y.append(1)  # Confused
            
            # Extract features for baseline words
            confused_times = []
            for events in confused_words.values():
                confused_times.extend([t for t, _ in events])
            confused_times = np.array(confused_times)
            
            baseline_count = 0
            for timestamp, word in all_words_timeline:
                if len(confused_times) > 0:
                    min_distance = np.min(np.abs(confused_times - timestamp))
                    if min_distance < 3.0:
                        continue
                
                if len(word) >= 3 and baseline_count < len(confused_words) * 2:
                    features = self._extract_features_batch(
                        timestamp, word, timestamps, eeg_processed, fnirs_processed
                    )
                    if features is not None:
                        all_X.append(features)
                        all_y.append(0)  # Not confused
                        baseline_count += 1
        
        if len(all_X) == 0:
            print("No training data extracted!")
            return False
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"\nTraining data summary:")
        print(f"  Total samples: {len(X)}")
        print(f"  Confused: {np.sum(y == 1)}")
        print(f"  Baseline: {np.sum(y == 0)}")
        
        # Train classifier
        print(f"\nTraining classifier...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate class weights
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights = np.array([class_weights[label] for label in y])
        
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=4,       # Reduced for speed
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.classifier.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Generate feature names
        self._generate_feature_names()
        
        print("✓ Classifier trained successfully!")
        return True
    
    def _preprocess_eeg_batch(self, eeg):
        """Preprocess EEG data for batch processing"""
        eeg_processed = eeg - np.mean(eeg, axis=0)
        
        # Bandpass filter
        nyquist = self.sample_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        if low < 1 and high < 1:
            b, a = signal.butter(4, [low, high], btype='band')
            for ch in range(eeg.shape[1]):
                if not np.all(np.isnan(eeg[:, ch])):
                    eeg_processed[:, ch] = signal.filtfilt(b, a, eeg[:, ch])
        
        return eeg_processed
    
    def _preprocess_fnirs_batch(self, fnirs):
        """Preprocess fNIRS data for batch processing"""
        if fnirs is None:
            return None
        
        fnirs_norm = fnirs[:, :4]  # First 4 are normalized
        
        # Apply lowpass filter
        nyquist = self.sample_rate / 2
        if 0.5 < nyquist:
            fnirs_low = 0.5 / nyquist
            b_low, a_low = signal.butter(4, fnirs_low, btype='low')
            
            for ch in range(fnirs_norm.shape[1]):
                if not np.all(np.isnan(fnirs_norm[:, ch])):
                    fnirs_norm[:, ch] = signal.detrend(fnirs_norm[:, ch])
                    fnirs_norm[:, ch] = signal.filtfilt(b_low, a_low, fnirs_norm[:, ch])
        
        return fnirs_norm
    
    def _extract_features_batch(self, timestamp, word, timestamps, eeg_processed, fnirs_processed):
        """Extract features for batch processing"""
        features = []
        
        # Find samples in window
        start_time = timestamp - self.prediction_window/2
        end_time = timestamp + self.prediction_window/2
        
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        window_indices = np.where(mask)[0]
        
        if len(window_indices) < 10:
            return None
        
        # EEG features
        eeg_window = eeg_processed[window_indices]
        
        for ch in range(eeg_window.shape[1]):
            channel_data = eeg_window[:, ch]
            
            # Time domain
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(np.abs(channel_data)),
                stats.skew(channel_data),
                stats.kurtosis(channel_data)
            ])
            
            # Frequency domain
            if len(channel_data) >= 64:
                freqs, psd = signal.welch(channel_data, fs=self.sample_rate, nperseg=64)
                
                bands = {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 50)
                }
                
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        band_power = np.mean(psd[band_mask])
                        features.append(np.log10(band_power + 1e-10))
                    else:
                        features.append(0)
            else:
                features.extend([0] * 5)
        
        # fNIRS features
        if fnirs_processed is not None:
            fnirs_window = fnirs_processed[window_indices]
            
            for ch in range(fnirs_window.shape[1]):
                channel_data = fnirs_window[:, ch]
                
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.max(channel_data) - np.min(channel_data)
                ])
        
        # Word features
        features.extend([
            len(word),
            self._count_syllables(word),
            1 if any(c.isdigit() for c in word) else 0
        ])
        
        return np.array(features)
    
    def extract_realtime_features(self, word):
        """Extract features for real-time prediction"""
        with self.lock:
            if len(self.timestamps) < self.sample_rate * self.prediction_window:
                return None
            
            features = []
            current_time = time.time()
            
            # Get recent data
            timestamps = np.array(list(self.timestamps))
            mask = timestamps >= (current_time - self.prediction_window)
            
            if np.sum(mask) < 10:
                return None
            
            # EEG features
            for ch in self.channels:
                if ch in self.eeg_processed and len(self.eeg_processed[ch]) > 0:
                    data = np.array(list(self.eeg_processed[ch]))
                    data = data[mask] if len(data) >= len(mask) else data
                    
                    if len(data) < 10:
                        features.extend([0] * 10)
                        continue
                    
                    # Time domain
                    features.extend([
                        np.mean(data),
                        np.std(data),
                        np.max(np.abs(data)),
                        stats.skew(data),
                        stats.kurtosis(data)
                    ])
                    
                    # Frequency domain
                    if len(data) >= 64:
                        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=64)
                        
                        bands = {
                            'delta': (0.5, 4),
                            'theta': (4, 8),
                            'alpha': (8, 13),
                            'beta': (13, 30),
                            'gamma': (30, 50)
                        }
                        
                        for band_name, (low, high) in bands.items():
                            band_mask = (freqs >= low) & (freqs <= high)
                            if np.any(band_mask):
                                band_power = np.mean(psd[band_mask])
                                features.append(np.log10(band_power + 1e-10))
                            else:
                                features.append(0)
                    else:
                        features.extend([0] * 5)
                else:
                    features.extend([0] * 10)
            
            # fNIRS features (simplified for real-time)
            fnirs_norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
            for ch in fnirs_norm_channels:
                if ch in self.fnirs_channels and len(self.fnirs_channels[ch]) > 0:
                    data = np.array(list(self.fnirs_channels[ch]))
                    data = data[mask] if len(data) >= len(mask) else data
                    
                    if len(data) > 0:
                        features.extend([
                            np.mean(data),
                            np.std(data),
                            np.max(data) - np.min(data)
                        ])
                    else:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0, 0, 0])
            
            # Word features
            features.extend([
                len(word),
                self._count_syllables(word),
                1 if any(c.isdigit() for c in word) else 0
            ])
            
            return np.array(features)
    
    def predict_confusion(self, word):
        """Predict confusion probability for a word"""
        if self.classifier is None or self.scaler is None:
            return 0.0
        
        features = self.extract_realtime_features(word)
        if features is None:
            return 0.0
        
        try:
            features_scaled = self.scaler.transform([features])
            prob = self.classifier.predict_proba(features_scaled)[0, 1]  # Probability of confusion
            return prob
        except:
            return 0.0
    
    def _count_syllables(self, word):
        """Simple syllable counter"""
        vowels = "aeiouAEIOU"
        count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        return max(1, count)
    
    def _generate_feature_names(self):
        """Generate feature names"""
        self.feature_names = []
        
        # EEG features per channel
        for ch in self.channels:
            self.feature_names.extend([
                f'EEG_{ch}_mean', f'EEG_{ch}_std', f'EEG_{ch}_max_abs',
                f'EEG_{ch}_skew', f'EEG_{ch}_kurtosis',
                f'EEG_{ch}_delta', f'EEG_{ch}_theta', f'EEG_{ch}_alpha',
                f'EEG_{ch}_beta', f'EEG_{ch}_gamma'
            ])
        
        # fNIRS features
        for i in range(4):
            self.feature_names.extend([
                f'fNIRS_Ch{i+1}_mean', f'fNIRS_Ch{i+1}_std', f'fNIRS_Ch{i+1}_range'
            ])
        
        # Word features
        self.feature_names.extend([
            'Word_length', 'Word_syllables', 'Word_has_number'
        ])
    
    def parse_osc_string(self, data, offset):
        """Parse null-terminated, 4-byte aligned string from OSC data"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        offset = ((end + 4) // 4) * 4
        return string, offset
    
    def parse_osc_message(self, data):
        """Parse an OSC message"""
        try:
            offset = 0
            
            address, offset = self.parse_osc_string(data, offset)
            if not address:
                return None
            
            if not address.startswith('/'):
                address = '/' + address
            
            type_tags, offset = self.parse_osc_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            type_tags = type_tags[1:]
            
            args = []
            for tag in type_tags:
                if tag == 'f':
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 'i':
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>i', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
            
            return {'address': address, 'args': args, 'type_tags': type_tags}
            
        except Exception as e:
            return None
    
    def process_osc_message(self, message):
        """Process incoming OSC message"""
        address = message['address']
        args = message['args']
        timestamp = time.time()
        
        with self.lock:
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                # EEG data
                if data_type == 'eeg' and len(args) == 4:
                    self.timestamps.append(timestamp)
                    
                    channels = ['TP9', 'AF7', 'AF8', 'TP10']
                    for ch, val in zip(channels, args):
                        self.eeg_channels[ch].append(val)
                        
                        # Apply real-time preprocessing
                        if ch in self.eeg_processed:
                            # Simple high-pass filter (DC removal)
                            if len(self.eeg_channels[ch]) > 50:
                                recent_data = list(self.eeg_channels[ch])[-50:]
                                filtered_val = val - np.mean(recent_data)
                                self.eeg_processed[ch].append(filtered_val)
                            else:
                                self.eeg_processed[ch].append(val)
                    
                    self.eeg_packet_count += 1
                    self.last_eeg_data = args
                
                # fNIRS data
                elif data_type == 'optics' and len(args) == 8:
                    norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
                    raw_channels = ['Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw']
                    
                    for i, (ch, val) in enumerate(zip(norm_channels + raw_channels, args)):
                        self.fnirs_channels[ch].append(val)
                    
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                
                # Other data types (accelerometer, gyro, etc.)
                elif data_type == 'acc' and len(args) == 3:
                    channels = ['acc_x', 'acc_y', 'acc_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[:3] = args
                
                elif data_type == 'gyro' and len(args) == 3:
                    channels = ['gyro_x', 'gyro_y', 'gyro_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[3:] = args
                
                elif data_type == 'drlref' and len(args) >= 2:
                    self.ref_channels['DRL'].append(args[0])
                    self.ref_channels['REF'].append(args[1])
                    self.last_ref_data = args[:2]
    
    def receiver_loop(self):
        """Main UDP receiver loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                self.packet_count += 1
                
                message = self.parse_osc_message(data)
                if message:
                    self.process_osc_message(message)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def prediction_loop(self):
        """Background thread for real-time prediction"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if we should update predictions
                if (current_time - self.last_prediction_time) >= self.prediction_update_interval:
                    if self.current_word and len(self.current_word) >= 2:
                        # Predict confusion for current word
                        confusion_prob = self.predict_confusion(self.current_word)
                        
                        # Store prediction
                        self.word_predictions[self.current_word] = confusion_prob
                        
                        # Update UI if word is confusing
                        if confusion_prob > self.prediction_threshold:
                            if self.teleprompter and self.teleprompter.active:
                                self.teleprompter.highlight_word(self.current_word, confusion_prob)
                    
                    self.last_prediction_time = current_time
                
                time.sleep(0.05)  # 50ms sleep
                
            except Exception as e:
                print(f"Prediction error: {e}")
    
    def setup_visualization(self):
        """Setup matplotlib figure and axes"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        gs = GridSpec(6, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        # Main plots
        self.axes['eeg'] = self.fig.add_subplot(gs[0, 0])
        self.axes['spectral'] = self.fig.add_subplot(gs[1, 0])
        self.axes['fnirs'] = self.fig.add_subplot(gs[2, 0])
        self.axes['motion'] = self.fig.add_subplot(gs[3, 0])
        self.axes['gyro'] = self.fig.add_subplot(gs[4, 0])
        self.axes['ref'] = self.fig.add_subplot(gs[5, 0])
        
        # Info panel
        self.axes['info'] = self.fig.add_subplot(gs[:5, 1])
        
        # Configure axes
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name != 'info':
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        
        # Titles and labels
        self.axes['eeg'].set_title('EEG Channels (4 channels)', fontsize=14, color='#4ECDC4', pad=10)
        self.axes['eeg'].set_ylabel('Amplitude (µV)')
        
        self.axes['spectral'].set_title('Spectral Analysis', fontsize=14, color='#FFD93D', pad=10)
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        
        self.axes['fnirs'].set_title('fNIRS/Optics', fontsize=14, color='#E74C3C', pad=10)
        self.axes['fnirs'].set_ylabel('Intensity')
        
        self.axes['motion'].set_title('Accelerometer', fontsize=14, color='#96CEB4', pad=10)
        self.axes['motion'].set_ylabel('Acceleration (g)')
        
        self.axes['gyro'].set_title('Gyroscope', fontsize=14, color='#9B59B6', pad=10)
        self.axes['gyro'].set_ylabel('Angular velocity (°/s)')
        
        self.axes['ref'].set_title('Reference Electrodes', fontsize=12, color='#95A5A6', pad=10)
        self.axes['ref'].set_ylabel('Voltage')
        self.axes['ref'].set_xlabel('Time (s)')
        
        # Info panel setup
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        self.axes['info'].set_title('Real-time Analysis', fontsize=14, color='#FFD93D', pad=10)
        
        # Initialize plot lines
        self._initialize_lines()
        
        plt.tight_layout()
    
    def _initialize_lines(self):
        """Initialize all plot lines"""
        # EEG lines
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95)
            self.lines[f'eeg_{ch}'] = line
        
        # Spectral lines
        self.spectral_lines = {}
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color, 
                                              linewidth=1.5, alpha=0.9)
            self.spectral_lines[ch] = line
        
        # fNIRS lines
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color, linewidth=1.5, alpha=0.9)
            self.lines[f'fnirs_{ch_norm}'] = line
        
        # Motion lines
        acc_colors = ['#3498DB', '#2ECC71', '#9B59B6']
        for i, ch in enumerate(['acc_x', 'acc_y', 'acc_z']):
            line, = self.axes['motion'].plot([], [], label=ch.replace('acc_', ''),
                                           color=acc_colors[i], linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Gyroscope lines
        gyro_colors = ['#F39C12', '#E67E22', '#D35400']
        for i, ch in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            line, = self.axes['gyro'].plot([], [], label=ch.replace('gyro_', ''),
                                         color=gyro_colors[i], linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Reference lines
        ref_colors = ['#95A5A6', '#7F8C8D']
        for i, ch in enumerate(['DRL', 'REF']):
            line, = self.axes['ref'].plot([], [], label=ch,
                                        color=ref_colors[i], linewidth=1.5, alpha=0.9)
            self.lines[f'ref_{ch}'] = line
        
        # Add legends
        self.axes['eeg'].legend(loc='upper left', fontsize=8, ncol=4, framealpha=0.7)
        self.axes['spectral'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        self.axes['fnirs'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.5)
        self.axes['motion'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['gyro'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['ref'].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Get minimum length
            min_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                           if len(self.eeg_channels[ch]) > 0)
            
            if min_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Calculate time axis
            timestamps = np.array(list(self.timestamps)[-min_length:])
            time_axis = timestamps - timestamps[-1]
            
            display_mask = time_axis >= -self.window_duration
            display_samples = np.sum(display_mask)
            
            # Update EEG
            for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_length:])
                        self.lines[line_key].set_data(time_axis[display_mask], 
                                                     data_array[display_mask])
            
            # Update other plots (fNIRS, motion, etc.) similarly...
            # [Similar code as in original file]
            
            # Update axes limits
            for ax in [self.axes['eeg'], self.axes['fnirs'], 
                      self.axes['motion'], self.axes['gyro'], self.axes['ref']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            # Update info panel
            self._update_info_panel()
        
        return list(self.lines.values()) + list(self.spectral_lines.values())
    
    def _update_info_panel(self):
        """Update statistics panel with real-time analysis"""
        self.axes['info'].clear()
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        y_pos = 0.95
        self.axes['info'].text(0.1, y_pos, 'Real-time Analysis', 
                             fontsize=12, weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
        # ML Status
        y_pos -= 0.08
        if self.classifier is not None:
            self.axes['info'].text(0.1, y_pos, '✓ ML Model Active', 
                                 fontsize=10, color='#4ECDC4',
                                 transform=self.axes['info'].transAxes)
        else:
            self.axes['info'].text(0.1, y_pos, '✗ No Model Loaded', 
                                 fontsize=10, color='#ff6666',
                                 transform=self.axes['info'].transAxes)
        
        # Current word and prediction
        y_pos -= 0.06
        if self.current_word:
            prob = self.word_predictions.get(self.current_word, 0.0)
            color = '#ff6666' if prob > self.prediction_threshold else '#66ff66'
            
            self.axes['info'].text(0.1, y_pos, f'Word: {self.current_word[:15]}', 
                                 fontsize=9, color='white',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'Confusion: {prob:.2%}', 
                                 fontsize=9, color=color, weight='bold',
                                 transform=self.axes['info'].transAxes)
        else:
            self.axes['info'].text(0.1, y_pos, 'Word: -', 
                                 fontsize=9, color='#aaaaaa',
                                 transform=self.axes['info'].transAxes)
        
        # Detection stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'Detections:', 
                             fontsize=10, weight='bold', color='#E74C3C',
                             transform=self.axes['info'].transAxes)
        
        y_pos -= 0.05
        confused_count = sum(1 for p in self.word_predictions.values() 
                           if p > self.prediction_threshold)
        total_words = len(self.word_predictions)
        
        self.axes['info'].text(0.15, y_pos, f'Confused: {confused_count}', 
                             fontsize=9, color='#ff6666',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        self.axes['info'].text(0.15, y_pos, f'Total: {total_words}', 
                             fontsize=9, color='white',
                             transform=self.axes['info'].transAxes)
        
        if total_words > 0:
            y_pos -= 0.04
            self.axes['info'].text(0.15, y_pos, f'Rate: {confused_count/total_words:.1%}', 
                                 fontsize=9, color='#FFD93D',
                                 transform=self.axes['info'].transAxes)
        
        # Data stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'Data:', 
                             fontsize=10, weight='bold', color='#96CEB4',
                             transform=self.axes['info'].transAxes)
        
        y_pos -= 0.05
        self.axes['info'].text(0.15, y_pos, f'EEG: {self.eeg_packet_count}', 
                             fontsize=9, color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        self.axes['info'].text(0.15, y_pos, f'fNIRS: {self.fnirs_packet_count}', 
                             fontsize=9, color='#E74C3C',
                             transform=self.axes['info'].transAxes)
        
        # Threshold info
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, f'Threshold: {self.prediction_threshold:.0%}', 
                             fontsize=9, color='#aaaaaa',
                             transform=self.axes['info'].transAxes)
    
    def start(self, training_files=None):
        """Start the real-time confusion detector"""
        print("\n" + "="*60)
        print("   REAL-TIME CONFUSION DETECTOR")
        print("="*60)
        
        # Load training data if provided
        if training_files:
            success = self.load_training_data(training_files)
            if not success:
                print("Failed to load training data. Running in visualization mode only.")
                self.classifier = None
        else:
            print("\nNo training data provided. Running in visualization mode only.")
            print("To enable real-time detection, provide training NPZ files.")
            self.classifier = None
        
        print(f"\n📡 Starting OSC receiver on port {self.port}")
        print("\n🧠 REAL-TIME MODE ACTIVE")
        
        if self.classifier:
            print("\n✓ ML model loaded - confusion detection enabled")
            print(f"  Detection threshold: {self.prediction_threshold:.0%}")
            print("\nConfusing words will be highlighted in RED")
        else:
            print("\n✗ No ML model - visualization only")
        
        print("\n📖 CONTROLS:")
        print("  ←/→ = Previous/Next text")
        print("  Space = Mark current word as confusing (validation)")
        print("  +/- = Adjust detection threshold")
        print("  'q' = Quit")
        
        # Start UDP receiver
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            self.running = True
            
            receiver_thread = threading.Thread(target=self.receiver_loop)
            receiver_thread.daemon = True
            receiver_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start receiver: {e}")
            return
        
        # Start prediction thread if model loaded
        if self.classifier:
            prediction_thread = threading.Thread(target=self.prediction_loop)
            prediction_thread.daemon = True
            prediction_thread.start()
        
        # Setup visualization
        self.setup_visualization()
        
        # Create teleprompter window IN MAIN THREAD
        print("\n🖥️ Opening teleprompter window...")
        self.teleprompter = RealTimeTeleprompterWindow(self)
        
        # Keyboard shortcuts for matplotlib window
        def on_key(event):
            if event.key == 'q':
                print("\nQuitting...")
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.on_close()
                self.cleanup_on_exit()
                plt.close('all')
                self.stop()
            elif event.key == '+' or event.key == '=':
                self.prediction_threshold = min(self.prediction_threshold + 0.05, 0.95)
                print(f"Detection threshold: {self.prediction_threshold:.0%}")
            elif event.key == '-':
                self.prediction_threshold = max(self.prediction_threshold - 0.05, 0.05)
                print(f"Detection threshold: {self.prediction_threshold:.0%}")
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Handle window close
        def on_close(event):
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
            self.cleanup_on_exit()
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=40,  # 25 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("\n✅ Real-time detection started!")
        print("🖥️ Focus the teleprompter window to read text")
        print("\n" + "="*60)
        
        # Show matplotlib window first (non-blocking)
        plt.show(block=False)
        
        try:
            # Run Tkinter in main thread with periodic matplotlib updates
            def update_plots():
                if self.teleprompter and self.teleprompter.active:
                    plt.pause(0.001)  # Process matplotlib events
                    self.teleprompter.root.after(40, update_plots)  # 25 FPS
            
            # Start plot updates
            update_plots()
            
            # Start teleprompter update loop
            self.teleprompter.update_loop()
            
            # Run Tkinter mainloop in main thread
            self.teleprompter.root.mainloop()
            
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
            self.cleanup_on_exit()
            self.stop()
    
    def stop(self):
        """Stop the detector"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.on_close()
        
        print("\nReal-time detector stopped.")
    
    def next_text(self):
        """Move to next text"""
        self.current_text_index = (self.current_text_index + 1) % len(TRAINING_TEXTS)
        self.word_predictions.clear()  # Clear predictions for new text
    
    def previous_text(self):
        """Move to previous text"""
        self.current_text_index = (self.current_text_index - 1) % len(TRAINING_TEXTS)
        self.word_predictions.clear()  # Clear predictions for new text


class RealTimeTeleprompterWindow:
    """Teleprompter window with real-time confusion highlighting"""
    def __init__(self, parent_detector):
        self.parent = parent_detector
        self.root = tk.Tk()
        self.root.title("👁 REAL-TIME CONFUSION DETECTION")
        
        # Make window large
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        
        # Track if window is active
        self.active = True
        
        # Confusion highlighting
        self.highlighted_words = set()
        self.word_tags = {}  # word -> list of text widget positions
        
        # Cursor tracking
        self.current_word = ""
        self.cursor_update_interval = 50
        self.last_cursor_update = 0
        
        # Header frame
        header_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        # Title label
        title_label = tk.Label(header_frame, 
                               text="👁 REAL-TIME CONFUSION DETECTION",
                               font=('Arial', 24, 'bold'),
                               fg='#FFD93D',
                               bg='#1a1a1a')
        title_label.pack(pady=10)
        
        # Instructions label
        status_text = "ML Model Active - Confusing words highlighted in RED" if self.parent.classifier else "No ML Model - Visualization Only"
        instructions = tk.Label(header_frame,
                               text=status_text,
                               font=('Arial', 14),
                               fg='#4ECDC4' if self.parent.classifier else '#ff6666',
                               bg='#1a1a1a')
        instructions.pack()
        
        # Main text frame
        text_frame = tk.Frame(self.root, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text display
        self.text_display = tk.Text(text_frame,
                                    font=('Georgia', 28, 'normal'),
                                    bg='#0a0a0a',
                                    fg='white',
                                    wrap=tk.WORD,
                                    padx=40,
                                    pady=30,
                                    spacing1=10,
                                    spacing2=8,
                                    spacing3=10,
                                    insertwidth=0,
                                    highlightthickness=0,
                                    borderwidth=0,
                                    relief=tk.FLAT,
                                    cursor="hand2")
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for highlighting
        self.text_display.tag_config('confused', 
                                    background='#661111',
                                    foreground='#ff6666',
                                    font=('Georgia', 28, 'bold'))
        
        self.text_display.tag_config('highly_confused', 
                                    background='#991111',
                                    foreground='#ffaaaa',
                                    font=('Georgia', 28, 'bold'))
        
        self.text_display.tag_config('manual_confused',
                                    background='#111166',
                                    foreground='#aaaaff',
                                    font=('Georgia', 28, 'bold'))
        
        # Make text read-only
        self.text_display.config(state=tk.DISABLED)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=120)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)
        
        # Status labels
        self.text_status = tk.Label(status_frame,
                                   text=f"Text: 1/{len(TRAINING_TEXTS)}",
                                   font=('Arial', 16),
                                   fg='#96CEB4',
                                   bg='#1a1a1a')
        self.text_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.detection_status = tk.Label(status_frame,
                                        text="Detected: 0 words",
                                        font=('Arial', 16),
                                        fg='#E74C3C',
                                        bg='#1a1a1a')
        self.detection_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.confusion_prob_label = tk.Label(status_frame,
                                           text="Current: -",
                                           font=('Arial', 16),
                                           fg='#FFD93D',
                                           bg='#1a1a1a')
        self.confusion_prob_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Threshold label
        self.threshold_label = tk.Label(status_frame,
                                       text=f"Threshold: {self.parent.prediction_threshold:.0%}",
                                       font=('Arial', 14),
                                       fg='#888888',
                                       bg='#1a1a1a')
        self.threshold_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Navigation hints
        nav_label = tk.Label(status_frame,
                           text="↑/↓: Scroll | ←/→: Change Text | Space: Mark Confusion | +/-: Threshold",
                           font=('Arial', 12),
                           fg='#888888',
                           bg='#1a1a1a')
        nav_label.pack(side=tk.BOTTOM, padx=20, pady=5)
        
        # Bind events
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.text_display.bind('<Motion>', self.on_mouse_motion)
        self.text_display.bind('<Leave>', self.on_mouse_leave)
        self.text_display.bind('<Button-1>', self.on_click)  # For manual marking
        
        # Update display
        self.update_display()
        
        # Start cursor tracking
        self.track_cursor()
        
        # Start highlighting update loop
        self.update_highlighting()
    
    def highlight_word(self, word, probability):
        """Highlight a word based on confusion probability"""
        if word not in self.highlighted_words or probability > self.parent.prediction_threshold:
            self.highlighted_words.add(word)
            
            # Find and highlight all occurrences
            self.text_display.config(state=tk.NORMAL)
            
            # Remove old highlighting for this word
            if word in self.word_tags:
                for tag in self.word_tags[word]:
                    self.text_display.tag_remove(tag, '1.0', tk.END)
            
            # Add new highlighting
            start_pos = '1.0'
            tag_name = 'highly_confused' if probability > 0.7 else 'confused'
            word_tag_list = []
            
            while True:
                pos = self.text_display.search(word, start_pos, tk.END, nocase=True)
                if not pos:
                    break
                
                end_pos = f"{pos}+{len(word)}c"
                unique_tag = f"{tag_name}_{word}_{pos}"
                self.text_display.tag_add(unique_tag, pos, end_pos)
                self.text_display.tag_config(unique_tag, 
                                           background='#991111' if probability > 0.7 else '#661111',
                                           foreground='#ffaaaa' if probability > 0.7 else '#ff6666',
                                           font=('Georgia', 28, 'bold'))
                word_tag_list.append(unique_tag)
                start_pos = end_pos
            
            self.word_tags[word] = word_tag_list
            self.text_display.config(state=tk.DISABLED)
    
    def on_mouse_motion(self, event):
        """Track mouse movement"""
        current_time = time.time() * 1000
        
        if current_time - self.last_cursor_update < self.cursor_update_interval:
            return
        
        self.last_cursor_update = current_time
        
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            word_start = self.text_display.index(f"{index} wordstart")
            word_end = self.text_display.index(f"{index} wordend")
            word = self.text_display.get(word_start, word_end).strip()
            
            if word and word != self.current_word:
                self.current_word = word
                self.parent.current_word = word
                
                # Update confusion probability display
                if word in self.parent.word_predictions:
                    prob = self.parent.word_predictions[word]
                    color = '#ff6666' if prob > self.parent.prediction_threshold else '#66ff66'
                    self.confusion_prob_label.config(
                        text=f"Current: {prob:.0%}",
                        fg=color
                    )
                else:
                    self.confusion_prob_label.config(
                        text="Current: ...",
                        fg='#aaaaaa'
                    )
                
        except:
            pass
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving"""
        self.current_word = ""
        self.parent.current_word = ""
        self.confusion_prob_label.config(text="Current: -", fg='#aaaaaa')
    
    def on_click(self, event):
        """Handle manual confusion marking"""
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            word_start = self.text_display.index(f"{index} wordstart")
            word_end = self.text_display.index(f"{index} wordend")
            word = self.text_display.get(word_start, word_end).strip().lower()
            
            if word:
                self.parent.manual_confusion_words.add(word)
                
                # Highlight manually marked word
                self.text_display.config(state=tk.NORMAL)
                self.text_display.tag_add('manual_confused', word_start, word_end)
                self.text_display.config(state=tk.DISABLED)
                
                print(f"✓ Manually marked '{word}' as confusing")
        except:
            pass
    
    def track_cursor(self):
        """Regular cursor tracking"""
        if self.active:
            try:
                x, y = self.text_display.winfo_pointerxy()
                widget_x = self.text_display.winfo_rootx()
                widget_y = self.text_display.winfo_rooty()
                
                rel_x = x - widget_x
                rel_y = y - widget_y
                
                if (0 <= rel_x <= self.text_display.winfo_width() and 
                    0 <= rel_y <= self.text_display.winfo_height()):
                    event = type('obj', (object,), {'x': rel_x, 'y': rel_y})
                    self.on_mouse_motion(event)
            except:
                pass
            
            self.root.after(50, self.track_cursor)
    
    def update_highlighting(self):
        """Update highlighting based on predictions"""
        if self.active:
            # Update detection count
            detected_count = len([w for w, p in self.parent.word_predictions.items() 
                                if p > self.parent.prediction_threshold])
            self.detection_status.config(text=f"Detected: {detected_count} words")
            
            # Update threshold display
            self.threshold_label.config(text=f"Threshold: {self.parent.prediction_threshold:.0%}")
            
            self.root.after(500, self.update_highlighting)  # Update every 500ms
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.keysym == 'Up':
            self.text_display.yview_scroll(-1, "units")
        elif event.keysym == 'Down':
            self.text_display.yview_scroll(1, "units")
        elif event.keysym == 'Left':
            self.parent.previous_text()
            self.update_display()
        elif event.keysym == 'Right':
            self.parent.next_text()
            self.update_display()
        elif event.keysym == 'space':
            # Mark current word as confusing
            if self.current_word:
                self.parent.manual_confusion_words.add(self.current_word.lower())
                print(f"✓ Marked '{self.current_word}' as confusing")
    
    def update_display(self):
        """Update text display"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        
        current_text = TRAINING_TEXTS[self.parent.current_text_index]
        self.text_display.insert('1.0', current_text)
        
        self.text_display.config(state=tk.DISABLED)
        self.text_display.yview_moveto(0)
        
        self.text_status.config(text=f"Text: {self.parent.current_text_index + 1}/{len(TRAINING_TEXTS)}")
        
        # Clear highlighted words for new text
        self.highlighted_words.clear()
        self.word_tags.clear()
    
    def update_status(self):
        """Update status display"""
        pass  # Status is updated in update_highlighting()
    
    def on_close(self):
        """Handle window close"""
        self.active = False
        self.root.destroy()
    
    def update_loop(self):
        """Regular update loop"""
        if self.active:
            self.update_status()
            self.root.after(100, self.update_loop)


def main():
    """Main function"""
    # Find training NPZ files
    search_paths = [
        "*confusion_clicks*.npz",
        "~/Downloads/*confusion_clicks*.npz",
        os.path.expanduser("~/Downloads/*confusion_clicks*.npz")
    ]
    
    npz_files = []
    for path in search_paths:
        npz_files.extend(glob.glob(path))
    
    if npz_files:
        print("Available training files:")
        for i, f in enumerate(npz_files):
            size = os.path.getsize(f) / 1024
            print(f"  {i+1}. {os.path.basename(f)} ({size:.1f} KB)")
        
        print("\nSelect training files (comma-separated numbers, or 'all'):")
        choice = input("Choice: ").strip()
        
        if choice.lower() == 'all':
            training_files = npz_files
        else:
            try:
                indices = [int(x.strip())-1 for x in choice.split(',')]
                training_files = [npz_files[i] for i in indices]
            except:
                print("Invalid selection. Running without training data.")
                training_files = None
    else:
        print("No training files found. Running in visualization mode.")
        training_files = None
    
    # Create and start detector
    detector = RealTimeConfusionDetector(
        port=8052,
        buffer_size=2000,
        window_duration=10
    )
    
    try:
        detector.start(training_files=training_files)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    print(f"\nRUNNING: Real-time Confusion Detector v2")
    print(f"Arguments: {sys.argv}")
    
    # If data file is provided as argument, use it
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"✓ Using training file: {sys.argv[1]}")
        training_files = [sys.argv[1]]
    else:
        print("✗ No valid training file provided")
        training_files = None
    
    # Create and start detector
    detector = RealTimeConfusionDetector(
        port=8052,
        buffer_size=2000,
        window_duration=10
    )
    
    detector.start(training_files=training_files)