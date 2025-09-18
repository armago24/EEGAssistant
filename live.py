#!/usr/bin/env python3
"""
Real-time EEG/fNIRS Confusion Detection System
Combines trained neural network with live data visualization
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, defaultdict
from datetime import datetime
import threading
import time
import sys
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import font as tkFont

# Import necessary components from the original files
import socket
import struct
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings('ignore')

# Import the neural network classes from the training script
# (These would normally be imported, but we'll redefine them here)
class ChannelAttention(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_channels, n_channels // 2)
        self.fc2 = torch.nn.Linear(n_channels // 2, n_channels)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=2)
        attn = F.relu(self.fc1(avg_pool))
        attn = torch.sigmoid(self.fc2(attn))
        return x * attn.unsqueeze(2)


class TemporalConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, dilation=dilation, 
                            padding=(kernel_size-1)//2)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleEEGEncoder(torch.nn.Module):
    def __init__(self, n_channels=4, n_timepoints=512):
        super().__init__()
        self.conv_3 = TemporalConvBlock(n_channels, 32, kernel_size=3)
        self.conv_5 = TemporalConvBlock(n_channels, 32, kernel_size=5)
        self.conv_7 = TemporalConvBlock(n_channels, 32, kernel_size=7)
        self.conv2 = TemporalConvBlock(96, 64, kernel_size=3, stride=2)
        self.conv3 = TemporalConvBlock(64, 128, kernel_size=3, stride=2)
        self.channel_attn = ChannelAttention(n_channels)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        
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


class fNIRSEncoder(torch.nn.Module):
    def __init__(self, n_channels=8, n_timepoints=512):
        super().__init__()
        self.conv1 = TemporalConvBlock(n_channels, 16, kernel_size=15)
        self.conv2 = TemporalConvBlock(16, 32, kernel_size=11, stride=2)
        self.conv3 = TemporalConvBlock(32, 64, kernel_size=7, stride=2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x


class ConfusionDetectorNN(torch.nn.Module):
    def __init__(self, n_eeg_ch=4, n_fnirs_ch=8, n_motion_ch=6, 
                 n_features=100, n_classes=3, n_timepoints=512):
        super().__init__()
        
        self.eeg_encoder = MultiScaleEEGEncoder(n_eeg_ch, n_timepoints)
        self.fnirs_encoder = fNIRSEncoder(n_fnirs_ch, n_timepoints)
        
        self.motion_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(n_motion_ch, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        
        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        fusion_dim = 128 + 64 + 16 + 64
        
        self.fusion_attention = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, fusion_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_dim // 2, fusion_dim),
            torch.nn.Sigmoid()
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, n_classes)
        )
        
    def forward(self, eeg, fnirs, motion, features):
        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)
        motion_feat = self.motion_encoder(motion)
        feat_encoded = self.feature_encoder(features)
        
        fused = torch.cat([eeg_feat, fnirs_feat, motion_feat, feat_encoded], dim=1)
        attn_weights = self.fusion_attention(fused)
        fused = fused * attn_weights
        output = self.classifier(fused)
        
        return output, attn_weights


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
        
    def count_syllables(self, word):
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


# Training texts (same as in original)
TRAINING_TEXTS = [
"""Epineural cuff electrodes. Epineural cuff electrodes are the simplest of nerve interface designs, usually containing 2 or more electrodes that are insulated and wrap around the surface of the epineurium of the peripheral nerve.""",

"""To date this type of interface is the only used in the clinic. This approach elicits a low FBR, making them quite stable for chronic implantation because the technology relies on compound signals to and from the nerve.""",
]


class RealTimeConfusionDetector:
    """Real-time confusion detection system"""
    
    def __init__(self, model_path, port=8052):
        self.model_path = model_path
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.load_model()
        
        # Initialize visualizer components
        self.setup_network()
        self.setup_buffers()
        self.setup_visualization_params()
        
        # Real-time inference
        self.inference_buffer_size = 512  # 2 seconds at 256Hz
        self.inference_overlap = 256  # 1 second overlap
        self.last_inference_time = 0
        self.inference_interval = 0.5  # seconds between inferences
        
        # Word tracking
        self.word_confusion_scores = defaultdict(list)
        self.sentence_confusion_scores = defaultdict(list)
        self.word_timestamps = {}
        self.current_word = ""
        self.current_sentence_idx = 0
        
        # Visualization
        self.show_predictions = False
        self.confusion_threshold = 0.5
        
        # UI components
        self.teleprompter = None
        self.current_text_index = 0
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        try:
            # B) Keep weights_only=True, but allowlist the safe globals you actually need.
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            
            # Extract model configuration
            config = checkpoint['model_config']
            self.model = ConfusionDetectorNN(
                n_eeg_ch=config['n_eeg_ch'],
                n_fnirs_ch=config['n_fnirs_ch'],
                n_motion_ch=config['n_motion_ch'],
                n_features=config['n_features'],
                n_classes=config['n_classes'],
                n_timepoints=config['n_timepoints']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load preprocessing components
            self.scaler = checkpoint['scaler']
            self.complexity_analyzer = checkpoint['complexity_analyzer']
            self.sample_rate = checkpoint['sample_rate']
            self.window_samples = config['n_timepoints']
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def setup_network(self):
        """Setup UDP receiver"""
        self.socket = None
        self.running = False
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
    def setup_buffers(self):
        """Setup data buffers"""
        self.buffer_size = 2000
        self.timestamps = deque(maxlen=self.buffer_size)
        
        self.eeg_channels = {ch: deque(maxlen=self.buffer_size) 
                            for ch in ['TP9', 'AF7', 'AF8', 'TP10']}
        self.fnirs_channels = {f'Ch{i}_{t}': deque(maxlen=self.buffer_size) 
                              for i in range(1,5) for t in ['norm', 'raw']}
        self.motion_channels = {ch: deque(maxlen=self.buffer_size) 
                               for ch in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
        
        self.lock = threading.Lock()
        
        # Filtered data
        self.eeg_filtered = None
        self.fnirs_filtered = None
        
        # Last values for interpolation
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        
    def setup_visualization_params(self):
        """Setup visualization parameters"""
        self.window_duration = 10
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.spectral_lines = {}
        
        # Colors
        self.eeg_colors = {'TP9': '#FF6B6B', 'AF7': '#4ECDC4', 
                          'AF8': '#45B7D1', 'TP10': '#96CEB4'}
        self.fnirs_colors = {f'Ch{i}': c for i, c in 
                            zip(range(1,5), ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])}
        
        self.spectral_window_size = 512
        self.max_freq = 70
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
    
    def parse_osc_message(self, data):
        """Parse OSC message from binary data"""
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
    
    def process_osc_message(self, message):
        """Process incoming OSC message"""
        address = message['address']
        args = message['args']
        timestamp = time.time()
        
        with self.lock:
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                if data_type == 'eeg' and len(args) == 4:
                    self.timestamps.append(timestamp)
                    for ch, val in zip(['TP9', 'AF7', 'AF8', 'TP10'], args):
                        self.eeg_channels[ch].append(val)
                    self.eeg_packet_count += 1
                    self.last_eeg_data = args
                    
                    # Trigger inference if needed
                    self.check_inference_trigger(timestamp)
                    
                elif data_type == 'optics' and len(args) == 8:
                    for i, ch in enumerate([f'Ch{j}_{t}' for j in range(1,5) for t in ['norm', 'raw']]):
                        self.fnirs_channels[ch].append(args[i])
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                
                elif data_type == 'acc' and len(args) == 3:
                    for ch, val in zip(['acc_x', 'acc_y', 'acc_z'], args):
                        self.motion_channels[ch].append(val)
                    if not self.last_motion_data:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[:3] = args
                
                elif data_type == 'gyro' and len(args) == 3:
                    for ch, val in zip(['gyro_x', 'gyro_y', 'gyro_z'], args):
                        self.motion_channels[ch].append(val)
                    if not self.last_motion_data:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[3:] = args
    
    def check_inference_trigger(self, current_time):
        """Check if it's time to run inference"""
        if current_time - self.last_inference_time >= self.inference_interval:
            if len(self.timestamps) >= self.window_samples:
                # Run inference in separate thread to avoid blocking
                inference_thread = threading.Thread(
                    target=self.run_inference,
                    args=(current_time,)
                )
                inference_thread.daemon = True
                inference_thread.start()
                self.last_inference_time = current_time
    
    def run_inference(self, timestamp):
        """Run model inference on current buffer"""
        try:
            # Extract window of data
            with self.lock:
                # Get aligned data
                min_len = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels)
                if min_len < self.window_samples:
                    return
                
                # Extract EEG
                eeg_data = np.array([
                    list(self.eeg_channels[ch])[-self.window_samples:]
                    for ch in ['TP9', 'AF7', 'AF8', 'TP10']
                ])
                
                # Extract fNIRS (normalized channels only)
                fnirs_data = np.array([
                    list(self.fnirs_channels[f'Ch{i}_norm'])[-self.window_samples:]
                    for i in range(1, 5)
                ])
                # Add raw channels
                fnirs_raw = np.array([
                    list(self.fnirs_channels[f'Ch{i}_raw'])[-self.window_samples:]
                    for i in range(1, 5)
                ])
                fnirs_data = np.vstack([fnirs_data, fnirs_raw])
                
                # Extract motion
                motion_data = np.array([
                    list(self.motion_channels[ch])[-self.window_samples:]
                    for ch in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                ])
                
                current_word = self.current_word
            
            # Preprocess signals
            eeg_filtered = self.preprocess_eeg(eeg_data)
            fnirs_filtered = self.preprocess_fnirs(fnirs_data)
            
            # Extract features
            word_features = self.complexity_analyzer.analyze_complexity(
                current_word if current_word else "default"
            )
            features = self.extract_features(eeg_filtered, fnirs_filtered, word_features)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Prepare tensors
            eeg_tensor = torch.FloatTensor(eeg_filtered).unsqueeze(0).to(self.device)
            fnirs_tensor = torch.FloatTensor(fnirs_filtered).unsqueeze(0).to(self.device)
            motion_tensor = torch.FloatTensor(motion_data).unsqueeze(0).to(self.device)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Run model
            with torch.no_grad():
                outputs, _ = self.model(eeg_tensor, fnirs_tensor, 
                                       motion_tensor, features_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Store results
            if current_word:
                self.word_confusion_scores[current_word].append(probabilities[1])
                self.word_timestamps[current_word] = timestamp
            
            # Detect sentence-level confusion
            sentence_confusion_prob = probabilities[2]
            if sentence_confusion_prob > self.confusion_threshold * 0.8:
                self.sentence_confusion_scores[self.current_sentence_idx].append(
                    sentence_confusion_prob
                )
            
            # Log high confusion events
            if probabilities[1] > self.confusion_threshold:
                print(f"🤔 Word confusion detected: '{current_word}' "
                      f"(prob={probabilities[1]:.2f})")
            elif probabilities[2] > self.confusion_threshold:
                print(f"📄 Sentence confusion detected "
                      f"(prob={probabilities[2]:.2f})")
            
        except Exception as e:
            print(f"Inference error: {e}")
    
    def preprocess_eeg(self, eeg_data):
        """Preprocess EEG data"""
        # Apply bandpass filter
        sos = signal.butter(4, [0.5, 50], btype='band', fs=self.sample_rate, output='sos')
        filtered = signal.sosfiltfilt(sos, eeg_data, axis=1)
        
        # Notch filters
        for freq in [60, 120]:
            sos_notch = signal.butter(4, [freq-2, freq+2], btype='bandstop', 
                                    fs=self.sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos_notch, filtered, axis=1)
        
        return filtered
    
    def preprocess_fnirs(self, fnirs_data):
        """Preprocess fNIRS data"""
        # Use normalized channels
        normalized = fnirs_data[:4]
        detrended = signal.detrend(normalized, axis=1)
        return fnirs_data  # Return all 8 channels
    
    def extract_features(self, eeg_window, fnirs_window, word_complexity):
        """Extract features for inference"""
        features = []
        
        # EEG features
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
            freqs, psd = signal.welch(ch_data, fs=self.sample_rate, 
                                     nperseg=min(256, len(ch_data)))
            
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
        
        # fNIRS features
        for ch in range(fnirs_window.shape[0]):
            ch_data = fnirs_window[ch]
            
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.max(ch_data) - np.min(ch_data),
                np.polyfit(np.arange(len(ch_data)), ch_data, 1)[0],
                np.argmax(ch_data) / len(ch_data)
            ])
        
        # Connectivity features
        def get_band_power(signal_data, band_name):
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate, 
                                     nperseg=min(256, len(signal_data)))
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
            low, high = bands[band_name]
            band_mask = (freqs >= low) & (freqs < high)
            return np.trapz(psd[band_mask], freqs[band_mask])
        
        af7_alpha = get_band_power(eeg_window[1], 'alpha')
        af8_alpha = get_band_power(eeg_window[2], 'alpha')
        features.append(af7_alpha - af8_alpha)
        
        # Inter-channel coherence
        channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for ch1, ch2 in channel_pairs:
            f, Cxy = signal.coherence(eeg_window[ch1], eeg_window[ch2], 
                                     fs=self.sample_rate, nperseg=min(64, len(eeg_window[ch1])))
            alpha_mask = (f >= 8) & (f <= 13)
            features.append(np.mean(Cxy[alpha_mask]))
        
        # Add word complexity features
        features.extend([
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
        ])
        
        return np.array(features)
    
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
    
    def setup_visualization(self):
        """Setup matplotlib visualization (same as original)"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid
        gs = GridSpec(6, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        # Create axes
        self.axes = {
            'eeg': self.fig.add_subplot(gs[0, 0]),
            'spectral': self.fig.add_subplot(gs[1, 0]),
            'fnirs': self.fig.add_subplot(gs[2, 0]),
            'motion': self.fig.add_subplot(gs[3, 0]),
            'gyro': self.fig.add_subplot(gs[4, 0]),
            'confusion': self.fig.add_subplot(gs[5, 0]),
            'info': self.fig.add_subplot(gs[:5, 1])
        }
        
        # Configure axes
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name not in ['info', 'confusion']:
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Titles
        titles = {
            'eeg': ('EEG Channels (4 channels)', '#4ECDC4'),
            'spectral': ('Spectral Analysis - Power Spectral Density', '#FFD93D'),
            'fnirs': ('fNIRS/Optics - Functional Near-Infrared Spectroscopy', '#E74C3C'),
            'motion': ('Accelerometer', '#96CEB4'),
            'gyro': ('Gyroscope', '#9B59B6'),
            'confusion': ('Real-time Confusion Detection', '#FF6B6B')
        }
        
        for ax_name, (title, color) in titles.items():
            self.axes[ax_name].set_title(title, fontsize=14, color=color, pad=10)
        
        # Labels
        self.axes['eeg'].set_ylabel('Amplitude (µV)')
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        self.axes['fnirs'].set_ylabel('Intensity')
        self.axes['motion'].set_ylabel('Acceleration (g)')
        self.axes['gyro'].set_ylabel('Angular velocity (°/s)')
        self.axes['confusion'].set_ylabel('Confusion Probability')
        self.axes['confusion'].set_xlabel('Time (s)')
        
        # Info panel
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        self.axes['info'].set_title('System Status', fontsize=14, color='#FFD93D', pad=10)
        
        # Initialize lines
        self._initialize_lines()
        
        # Add confusion probability line
        self.confusion_line, = self.axes['confusion'].plot([], [], 
                                                         color='#FF6B6B', 
                                                         linewidth=2)
        self.axes['confusion'].axhline(y=self.confusion_threshold, 
                                      color='yellow', linestyle='--', 
                                      alpha=0.5, label='Threshold')
        self.axes['confusion'].set_ylim(0, 1)
        self.axes['confusion'].legend()
        
        plt.tight_layout()
    
    def _initialize_lines(self):
        """Initialize plot lines (same as original)"""
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
        
        self.axes['spectral'].set_xlim(0, self.max_freq)
        
        # fNIRS lines
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color, linewidth=1.5, alpha=0.9)
            self.lines[f'fnirs_{ch_norm}'] = line
        
        # Motion lines
        colors = {
            'acc': ['#3498DB', '#2ECC71', '#9B59B6'],
            'gyro': ['#F39C12', '#E67E22', '#D35400']
        }
        
        for prefix, ax_name in [('acc', 'motion'), ('gyro', 'gyro')]:
            for i, axis in enumerate(['x', 'y', 'z']):
                ch = f'{prefix}_{axis}'
                line, = self.axes[ax_name].plot([], [], label=axis,
                                               color=colors[prefix][i],
                                               linewidth=1.5, alpha=0.9)
                self.lines[f'motion_{ch}'] = line
        
        # Add legends
        for ax_name in ['eeg', 'spectral', 'fnirs', 'motion', 'gyro']:
            self.axes[ax_name].legend(loc='upper right', fontsize=8, 
                                     ncol=4 if ax_name in ['eeg', 'spectral'] else 3,
                                     framealpha=0.5)
    
    def update_plot(self, frame):
        """Update all plots including confusion probabilities"""
        # Update regular plots (same as original)
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values()) + [self.confusion_line]
            
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values()) + [self.confusion_line]
            
            # Time axis
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                time_axis = timestamps - timestamps[-1]
                display_mask = time_axis >= -self.window_duration
                display_samples = np.sum(display_mask)
            else:
                return list(self.lines.values()) + list(self.spectral_lines.values()) + [self.confusion_line]
            
            # Update EEG
            filtered_eeg_data = {}
            eeg_values_for_scaling = []
            
            for ch_name in self.eeg_channels:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
                        if len(data_array) > 50:
                            window_size = min(50, len(data_array) // 4)
                            if window_size > 1:
                                moving_avg = np.convolve(data_array, 
                                                       np.ones(window_size)/window_size, 
                                                       mode='same')
                                filtered_data = data_array - moving_avg
                            else:
                                filtered_data = data_array - np.mean(data_array)
                        else:
                            filtered_data = data_array - np.mean(data_array)
                        
                        filtered_eeg_data[ch_name] = filtered_data
                        
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            # Update spectral analysis
            if len(filtered_eeg_data) == 4:
                all_psd_values = []
                
                for ch_name, data in filtered_eeg_data.items():
                    if ch_name in self.spectral_lines:
                        frequencies, psd = self.compute_spectrum(data[-int(self.sample_rate * 4):])
                        
                        if frequencies is not None:
                            self.spectral_lines[ch_name].set_data(frequencies, psd)
                            all_psd_values.extend(psd)
                
                if all_psd_values:
                    y_min = np.percentile(all_psd_values, 5) - 5
                    y_max = np.percentile(all_psd_values, 95) + 5
                    self.axes['spectral'].set_ylim(y_min, y_max)
            
            # Update other channels
            for channel_dict, prefix in [(self.fnirs_channels, 'fnirs'), 
                                        (self.motion_channels, 'motion')]:
                for ch_name, data_deque in channel_dict.items():
                    if len(data_deque) > 0:
                        if prefix == 'fnirs' and ch_name.endswith('_norm'):
                            line_key = f'{prefix}_{ch_name}'
                        elif prefix == 'fnirs' and ch_name.endswith('_raw'):
                            continue
                        else:
                            line_key = f'{prefix}_{ch_name}'
                        
                        if line_key in self.lines:
                            data_array = np.array(list(data_deque))
                            if len(data_array) >= len(display_mask):
                                aligned_data = data_array[-len(display_mask):]
                                self.lines[line_key].set_data(
                                    time_axis[display_mask],
                                    aligned_data[display_mask]
                                )
            
            # Update axes limits
            for ax in [self.axes['eeg'], self.axes['fnirs'], 
                      self.axes['motion'], self.axes['gyro']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            # Special EEG scaling
            if eeg_values_for_scaling:
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                y_range = 4 * eeg_std
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
            
            # Update confusion probability plot
            # (This would show recent confusion scores)
            self.axes['confusion'].set_xlim(-self.window_duration, 0)
            
            # Update info panel
            self._update_info_panel()
        
        return list(self.lines.values()) + list(self.spectral_lines.values()) + [self.confusion_line]
    
    def compute_spectrum(self, data, sample_rate=256):
        """Compute power spectral density"""
        if len(data) < self.spectral_window_size:
            return None, None
        
        frequencies, psd = signal.welch(
            data, 
            fs=sample_rate, 
            nperseg=min(len(data), self.spectral_window_size),
            noverlap=min(len(data)//2, self.spectral_window_size//2),
            scaling='density'
        )
        
        freq_mask = frequencies <= self.max_freq
        frequencies = frequencies[freq_mask]
        psd = psd[freq_mask]
        psd_db = 10 * np.log10(psd + 1e-10)
        
        return frequencies, psd_db
    
    def _update_info_panel(self):
        """Update statistics panel with confusion detection info"""
        self.axes['info'].clear()
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        y_pos = 0.95
        self.axes['info'].text(0.1, y_pos, 'System Status', 
                             fontsize=12, weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
        # Model status
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, '🧠 Model: Active', 
                             fontsize=10, color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        
        # Current word
        y_pos -= 0.05
        self.axes['info'].text(0.1, y_pos, f'Word: {self.current_word[:15] if self.current_word else "-"}', 
                             fontsize=9, color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
        # Confusion stats
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, 'Detected:', 
                             fontsize=10, weight='bold', color='#FF6B6B',
                             transform=self.axes['info'].transAxes)
        
        y_pos -= 0.04
        confused_words = sum(1 for word, scores in self.word_confusion_scores.items() 
                           if scores and max(scores) > self.confusion_threshold)
        self.axes['info'].text(0.1, y_pos, f'🤔 Words: {confused_words}', 
                             fontsize=9, color='#FF8866',
                             transform=self.axes['info'].transAxes)
        
        y_pos -= 0.04
        confused_sentences = sum(1 for idx, scores in self.sentence_confusion_scores.items() 
                               if scores and max(scores) > self.confusion_threshold)
        self.axes['info'].text(0.1, y_pos, f'📄 Sentences: {confused_sentences}', 
                             fontsize=9, color='#FF6655',
                             transform=self.axes['info'].transAxes)
        
        # Prediction visibility
        y_pos -= 0.05
        mode_text = "SHOWING" if self.show_predictions else "HIDDEN"
        mode_color = '#4ECDC4' if self.show_predictions else '#888888'
        self.axes['info'].text(0.1, y_pos, f'Predictions: {mode_text}', 
                             fontsize=9, color=mode_color,
                             transform=self.axes['info'].transAxes)
        
        # Text info
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, f'Text: {self.current_text_index + 1}/{len(TRAINING_TEXTS)}', 
                             fontsize=9, color='#aaaaaa',
                             transform=self.axes['info'].transAxes)
        
        # EEG stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'EEG (4ch):', fontsize=10, 
                             weight='bold', color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        for ch in self.eeg_channels:
            if len(self.eeg_channels[ch]) > 0:
                data = np.array(list(self.eeg_channels[ch])[-100:])
                mean_val = np.mean(data)
                std_val = np.std(data)
                self.axes['info'].text(0.15, y_pos, f'{ch}:', fontsize=9,
                                     color=self.eeg_colors[ch],
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.4, y_pos, f'{mean_val:.1f}±{std_val:.1f}',
                                     fontsize=9, color='white',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.04
        
        # System stats
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, 'Packets:', fontsize=10,
                             weight='bold', color='#95A5A6',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        stats = [
            ('Total:', self.packet_count),
            ('EEG:', self.eeg_packet_count),
            ('fNIRS:', self.fnirs_packet_count)
        ]
        
        for label, value in stats:
            self.axes['info'].text(0.15, y_pos, label, fontsize=9, color='white',
                                 transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.4, y_pos, f'{value}', fontsize=9, color='white',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
    
    def start(self):
        """Start the real-time confusion detection system"""
        print("\n" + "="*60)
        print("   REAL-TIME EEG/fNIRS CONFUSION DETECTION SYSTEM")
        print("="*60)
        print(f"\n🧠 Model loaded on {self.device}")
        print(f"📡 Listening for OSC data on UDP port {self.port}")
        print("\n⌨️  Controls:")
        print("  • D: Toggle confusion highlighting")
        print("  • ←/→: Change text")
        print("  • +/-: Adjust threshold")
        print("  • Q: Quit")
        
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
        
        # Setup visualization
        self.setup_visualization()
        
        # Create enhanced teleprompter
        print("\n🖥️ Opening enhanced teleprompter window...")
        self.teleprompter = EnhancedTeleprompterWindow(self)
        self.teleprompter.update_loop()
        
        # Keyboard shortcuts
        def on_key(event):
            if event.key == 'q':
                print("\nQuitting...")
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.on_close()
                plt.close('all')
                self.stop()
            elif event.key == 'd':
                self.toggle_predictions()
            elif event.key in ['+', '=']:
                self.confusion_threshold = min(self.confusion_threshold + 0.05, 0.95)
                print(f"Confusion threshold: {self.confusion_threshold:.2f}")
            elif event.key == '-':
                self.confusion_threshold = max(self.confusion_threshold - 0.05, 0.05)
                print(f"Confusion threshold: {self.confusion_threshold:.2f}")
            elif event.key == 'Left':
                self.previous_text()
            elif event.key == 'Right':
                self.next_text()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=40,  # 25 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("\n✅ System started! Read the text and press 'D' to see detected confusion.")
        print("\n" + "="*60)
        
        try:
            # Run both windows
            def run_teleprompter():
                if self.teleprompter:
                    self.teleprompter.root.mainloop()
            
            teleprompter_thread = threading.Thread(target=run_teleprompter)
            teleprompter_thread.daemon = True
            teleprompter_thread.start()
            
            plt.show()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
        finally:
            self.stop()
    
    def toggle_predictions(self):
        """Toggle visibility of confusion predictions"""
        self.show_predictions = not self.show_predictions
        if self.teleprompter:
            self.teleprompter.update_display()
        
        mode = "visible" if self.show_predictions else "hidden"
        print(f"\n👁️ Confusion predictions: {mode}")
    
    def next_text(self):
        """Navigate to next text"""
        self.current_text_index = (self.current_text_index + 1) % len(TRAINING_TEXTS)
        self.reset_confusion_scores()
        if self.teleprompter:
            self.teleprompter.update_display()
        print(f"\n📖 Text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
    
    def previous_text(self):
        """Navigate to previous text"""
        self.current_text_index = (self.current_text_index - 1) % len(TRAINING_TEXTS)
        self.reset_confusion_scores()
        if self.teleprompter:
            self.teleprompter.update_display()
        print(f"\n📖 Text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
    
    def reset_confusion_scores(self):
        """Reset confusion scores for new text"""
        self.word_confusion_scores.clear()
        self.sentence_confusion_scores.clear()
        self.word_timestamps.clear()
        self.current_sentence_idx = 0
    
    def stop(self):
        """Stop the system"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.on_close()
        
        print(f"\n{'='*50}")
        print(f"SYSTEM STOPPED")
        print(f"Total packets: {self.packet_count}")
        print(f"Confused words detected: {len(self.word_confusion_scores)}")
        print(f"{'='*50}\n")


class EnhancedTeleprompterWindow:
    """Enhanced teleprompter with real-time confusion highlighting"""
    
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("👁 REAL-TIME CONFUSION DETECTION")
        
        # Window setup
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        self.active = True
        
        # Cursor tracking
        self.current_word = ""
        self.cursor_update_interval = 50
        self.last_cursor_update = 0
        
        # Text tags for highlighting
        self.word_tags = {}
        self.sentence_tags = {}
        
        # UI Setup
        self._setup_ui()
        
        # Bind events
        self._bind_events()
        
        # Initialize display
        self.update_display()
        self.track_cursor()
    
    def _setup_ui(self):
        """Setup the UI components"""
        # Header
        header_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, 
                text="👁 REAL-TIME CONFUSION DETECTION",
                font=('Arial', 24, 'bold'),
                fg='#FFD93D',
                bg='#1a1a1a').pack(pady=10)
        
        tk.Label(header_frame,
                text="Read the text • Press D to toggle confusion highlighting",
                font=('Arial', 14),
                fg='#4ECDC4',
                bg='#1a1a1a').pack()
        
        # Text display
        text_frame = tk.Frame(self.root, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
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
        self.text_display.config(state=tk.DISABLED)
        
        # Configure highlight colors
        self.text_display.tag_configure("word_confusion", 
                                       background="#ff6666", 
                                       foreground="white",
                                       font=('Georgia', 28, 'bold'))
        self.text_display.tag_configure("sentence_confusion", 
                                       background="#ffaa66", 
                                       foreground="white")
        self.text_display.tag_configure("current_word",
                                       underline=True,
                                       underlinefg="#FFD93D")
        
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
        
        self.prediction_status = tk.Label(status_frame,
                                        text="🔍 Predictions: HIDDEN",
                                        font=('Arial', 16, 'bold'),
                                        fg='#888888',
                                        bg='#1a1a1a')
        self.prediction_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.confusion_status = tk.Label(status_frame,
                                       text="Confused: Words=0, Sentences=0",
                                       font=('Arial', 16),
                                       fg='#E74C3C',
                                       bg='#1a1a1a')
        self.confusion_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.threshold_status = tk.Label(status_frame,
                                       text=f"Threshold: {self.detector.confusion_threshold:.2f}",
                                       font=('Arial', 14),
                                       fg='#FFD93D',
                                       bg='#1a1a1a')
        self.threshold_status.pack(side=tk.RIGHT, padx=20, pady=5)
        
        # Additional status
        self.word_status = tk.Label(status_frame,
                                   text="Current word: -",
                                   font=('Arial', 14, 'italic'),
                                   fg='#FFD93D',
                                   bg='#1a1a1a')
        self.word_status.pack(side=tk.RIGHT, padx=20, pady=5)
        
        tk.Label(status_frame,
                text="D: Toggle Highlights | ←/→: Change Text | +/-: Threshold | Q: Quit",
                font=('Arial', 12),
                fg='#888888',
                bg='#1a1a1a').pack(side=tk.BOTTOM, padx=20, pady=5)
    
    def _bind_events(self):
        """Bind all event handlers"""
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Mouse events
        self.text_display.bind('<Motion>', self.on_mouse_motion)
        self.text_display.bind('<Leave>', self.on_mouse_leave)
    
    def on_mouse_motion(self, event):
        """Track cursor position over text and detect sentence"""
        current_time = time.time() * 1000
        
        if current_time - self.last_cursor_update < self.cursor_update_interval:
            return
        
        self.last_cursor_update = current_time
        
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            
            # Get word at cursor
            word_start = self.text_display.index(f"{index} wordstart")
            word_end = self.text_display.index(f"{index} wordend")
            word = self.text_display.get(word_start, word_end).strip()
            
            # Remove previous highlighting of current word
            self.text_display.tag_remove("current_word", "1.0", tk.END)
            
            if word and word != self.current_word:
                self.current_word = word
                self.detector.current_word = word
                self.word_status.config(text=f"Current word: {word}")
                
                # Highlight current word
                self.text_display.tag_add("current_word", word_start, word_end)
            
            # Detect current sentence
            text_content = self.text_display.get("1.0", tk.END)
            cursor_pos = len(self.text_display.get("1.0", index))
            
            # Find sentence boundaries
            sentences = text_content.split('.')
            char_count = 0
            for i, sentence in enumerate(sentences):
                if char_count <= cursor_pos < char_count + len(sentence) + 1:
                    self.detector.current_sentence_idx = i
                    break
                char_count += len(sentence) + 1
                
        except:
            pass
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving text area"""
        self.current_word = ""
        self.detector.current_word = ""
        self.word_status.config(text="Current word: -")
        self.text_display.tag_remove("current_word", "1.0", tk.END)
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.char.lower() == 'd':
            self.detector.toggle_predictions()
        elif event.char.lower() == 'q':
            self.on_close()
        elif event.keysym == 'Left':
            self.detector.previous_text()
        elif event.keysym == 'Right':
            self.detector.next_text()
        elif event.char in ['+', '=']:
            self.detector.confusion_threshold = min(self.detector.confusion_threshold + 0.05, 0.95)
            self.threshold_status.config(text=f"Threshold: {self.detector.confusion_threshold:.2f}")
        elif event.char == '-':
            self.detector.confusion_threshold = max(self.detector.confusion_threshold - 0.05, 0.05)
            self.threshold_status.config(text=f"Threshold: {self.detector.confusion_threshold:.2f}")
    
    def update_display(self):
        """Update text display with confusion highlighting"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        
        # Insert text
        current_text = TRAINING_TEXTS[self.detector.current_text_index]
        self.text_display.insert('1.0', current_text)
        
        # Apply confusion highlighting if enabled
        if self.detector.show_predictions:
            self.apply_confusion_highlighting()
        
        self.text_display.config(state=tk.DISABLED)
        self.text_display.yview_moveto(0)
        
        # Update status
        self.text_status.config(text=f"Text: {self.detector.current_text_index + 1}/{len(TRAINING_TEXTS)}")
        
        if self.detector.show_predictions:
            self.prediction_status.config(text="🔍 Predictions: VISIBLE", fg='#4ECDC4')
        else:
            self.prediction_status.config(text="🔍 Predictions: HIDDEN", fg='#888888')
        
        self.update_confusion_stats()
    
    def apply_confusion_highlighting(self):
        """Apply highlighting based on confusion scores"""
        text_content = self.text_display.get("1.0", tk.END)
        
        # Clear existing tags
        self.text_display.tag_remove("word_confusion", "1.0", tk.END)
        self.text_display.tag_remove("sentence_confusion", "1.0", tk.END)
        
        # Highlight confused words
        for word, scores in self.detector.word_confusion_scores.items():
            if scores and max(scores) > self.detector.confusion_threshold:
                # Find all occurrences of the word
                start_pos = "1.0"
                while True:
                    pos = self.text_display.search(word, start_pos, tk.END, nocase=True)
                    if not pos:
                        break
                    
                    # Check if it's a whole word
                    word_start = self.text_display.index(f"{pos} wordstart")
                    word_end = self.text_display.index(f"{pos} wordend")
                    found_word = self.text_display.get(word_start, word_end).strip()
                    
                    if found_word.lower() == word.lower():
                        self.text_display.tag_add("word_confusion", word_start, word_end)
                    
                    start_pos = word_end
        
        # Highlight confused sentences
        sentences = text_content.split('.')
        char_pos = "1.0"
        
        for i, sentence in enumerate(sentences):
            if i in self.detector.sentence_confusion_scores:
                scores = self.detector.sentence_confusion_scores[i]
                if scores and max(scores) > self.detector.confusion_threshold:
                    # Find sentence boundaries
                    sentence_start = char_pos
                    sentence_end = self.text_display.index(f"{char_pos} + {len(sentence) + 1} chars")
                    
                    # Apply sentence highlighting (lighter than word highlighting)
                    self.text_display.tag_add("sentence_confusion", sentence_start, sentence_end)
            
            # Move to next sentence
            char_pos = self.text_display.index(f"{char_pos} + {len(sentence) + 1} chars")
    
    def update_confusion_stats(self):
        """Update confusion statistics"""
        confused_words = sum(1 for word, scores in self.detector.word_confusion_scores.items() 
                           if scores and max(scores) > self.detector.confusion_threshold)
        confused_sentences = sum(1 for idx, scores in self.detector.sentence_confusion_scores.items() 
                               if scores and max(scores) > self.detector.confusion_threshold)
        
        self.confusion_status.config(
            text=f"Confused: Words={confused_words}, Sentences={confused_sentences}"
        )
    
    def track_cursor(self):
        """Track cursor position for word recording"""
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
    
    def on_close(self):
        """Clean window close"""
        self.active = False
        self.root.destroy()
    
    def update_loop(self):
        """Regular update loop"""
        if self.active:
            self.update_confusion_stats()
            self.threshold_status.config(text=f"Threshold: {self.detector.confusion_threshold:.2f}")
            
            # Re-apply highlighting if predictions are visible
            if self.detector.show_predictions:
                self.apply_confusion_highlighting()
            
            self.root.after(500, self.update_loop)  # Update every 500ms


def main():
    parser = argparse.ArgumentParser(
        description='Real-time EEG/fNIRS confusion detection system'
    )
    parser.add_argument(
        'model_path',
        help='Path to trained model (.pth file)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8052,
        help='UDP port for OSC data (default: 8052)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Initial confusion detection threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check model file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create detector
    detector = RealTimeConfusionDetector(
        model_path=args.model_path,
        port=args.port
    )
    detector.confusion_threshold = args.threshold
    
    # Start system
    try:
        detector.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()