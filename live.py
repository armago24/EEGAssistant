"""
===============================================================================
LIVE.PY - REAL-TIME CONFUSION DETECTION SYSTEM (FIXED OSC RECEPTION)
===============================================================================
An end-to-end system that combines EEG/fNIRS data collection with neural network
inference to predict reading confusion in real-time.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
import torch
import socket
import struct
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Feature extraction utilities
def extract_word_complexity_features(word):
    """Extract complexity features from a word."""
    if not word:
        return np.zeros(14)
    
    features = []
    features.append(len(word))  # word_length
    
    # Estimate syllables
    vowels = sum(1 for c in word.lower() if c in 'aeiou')
    features.append(max(1, vowels))  # word_syllables
    
    features.append(len(set(word)))  # word_unique_chars
    features.append(len(set(word)) / len(word) if len(word) > 0 else 0)  # word_char_variety_ratio
    
    vowel_count = sum(1 for c in word.lower() if c in 'aeiou')
    consonant_count = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
    features.append(vowel_count)  # word_vowel_count
    features.append(consonant_count)  # word_consonant_count
    features.append(vowel_count / (consonant_count + 1))  # word_vowel_consonant_ratio
    
    # Check for double letters
    has_double = any(word[i] == word[i+1] for i in range(len(word)-1) if i < len(word)-1)
    features.append(1 if has_double else 0)  # word_has_double_letters
    
    features.append(1 if any(c.isupper() for c in word) else 0)  # word_has_capital
    
    # Difficult patterns
    difficult_patterns = ['ph', 'gh', 'tion', 'sion', 'ough', 'augh']
    pattern_count = sum(1 for pattern in difficult_patterns if pattern in word.lower())
    features.append(pattern_count)  # word_difficult_pattern_count
    
    # Medical/technical terms (simplified)
    medical_prefixes = ['bio', 'neuro', 'cardio', 'hemo', 'patho', 'micro']
    is_medical = any(word.lower().startswith(prefix) for prefix in medical_prefixes)
    features.append(1 if is_medical else 0)  # word_is_medical_term
    
    # Prefix/suffix counts
    common_prefixes = ['un', 're', 'dis', 'pre', 'mis', 'over', 'under', 'non']
    prefix_count = sum(1 for prefix in common_prefixes if word.lower().startswith(prefix))
    features.append(prefix_count)  # word_prefix_count
    
    common_suffixes = ['ing', 'ed', 'ly', 'ness', 'ment', 'ful', 'less', 'tion']
    suffix_count = sum(1 for suffix in common_suffixes if word.lower().endswith(suffix))
    features.append(suffix_count)  # word_suffix_count
    
    # Estimated grade level (simplified)
    grade_level = min(12, len(word) / 2 + vowels - 1)
    features.append(grade_level)  # word_estimated_grade_level
    
    return np.array(features)


class ConfusionDetectorModel:
    """Wrapper for the trained confusion detection model."""
    
    def __init__(self, model_path):
        self.model = None
        self.scaler = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model from file."""
        try:
            # Load model package
            model_package = torch.load(model_path, map_location=self.device)
            
            # Extract components
            self.scaler = model_package['scaler']
            self.model_config = model_package['model_config']
            
            # Create model architecture (simplified version)
            from torch import nn
            
            class SimplifiedConfusionDetector(nn.Module):
                def __init__(self, n_eeg_ch, n_fnirs_ch, n_motion_ch, n_features, n_classes, n_timepoints):
                    super().__init__()
                    
                    # Convolutional layers for time series
                    self.eeg_conv = nn.Conv1d(n_eeg_ch, 32, kernel_size=5, padding=2)
                    self.fnirs_conv = nn.Conv1d(n_fnirs_ch, 16, kernel_size=5, padding=2)
                    self.motion_conv = nn.Conv1d(n_motion_ch, 16, kernel_size=5, padding=2)
                    
                    # Fusion layer
                    fusion_size = 32 + 16 + 16 + n_features
                    self.fusion = nn.Sequential(
                        nn.Linear(fusion_size, 128),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, n_classes)
                    )
                    
                    # Attention weights
                    self.attention = nn.Linear(fusion_size, fusion_size)
                
                def forward(self, eeg, fnirs, motion, features):
                    # Process time series
                    eeg_feat = torch.relu(self.eeg_conv(eeg))
                    eeg_feat = torch.mean(eeg_feat, dim=2)
                    
                    fnirs_feat = torch.relu(self.fnirs_conv(fnirs))
                    fnirs_feat = torch.mean(fnirs_feat, dim=2)
                    
                    motion_feat = torch.relu(self.motion_conv(motion))
                    motion_feat = torch.mean(motion_feat, dim=2)
                    
                    # Concatenate all features
                    combined = torch.cat([eeg_feat, fnirs_feat, motion_feat, features], dim=1)
                    
                    # Apply attention
                    attention_weights = torch.softmax(self.attention(combined), dim=1)
                    combined = combined * attention_weights
                    
                    # Final classification
                    output = self.fusion(combined)
                    
                    return output, attention_weights
            
            # Initialize model
            self.model = SimplifiedConfusionDetector(
                n_eeg_ch=self.model_config['n_eeg_ch'],
                n_fnirs_ch=self.model_config['n_fnirs_ch'],
                n_motion_ch=self.model_config['n_motion_ch'],
                n_features=self.model_config['n_features'],
                n_classes=self.model_config['n_classes'],
                n_timepoints=self.model_config['n_timepoints']
            )
            
            # Load weights
            self.model.load_state_dict(model_package['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_signals(self, eeg_data, fnirs_data, motion_data):
        """Apply required preprocessing to signals."""
        # EEG preprocessing
        if len(eeg_data) > 10:
            # Bandpass filter 0.5-40 Hz
            b, a = signal.butter(4, [0.5, 40], btype='band', fs=256)
            eeg_filtered = signal.filtfilt(b, a, eeg_data, axis=0)
            
            # Notch filter at 60 Hz
            b_notch, a_notch = signal.iirnotch(60, 30, fs=256)
            eeg_filtered = signal.filtfilt(b_notch, a_notch, eeg_filtered, axis=0)
        else:
            eeg_filtered = eeg_data
        
        # fNIRS preprocessing
        if len(fnirs_data) > 10:
            # Low-pass filter at 0.5 Hz
            b, a = signal.butter(4, 0.5, btype='low', fs=256)
            fnirs_filtered = signal.filtfilt(b, a, fnirs_data, axis=0)
            
            # Detrend
            fnirs_filtered = signal.detrend(fnirs_filtered, axis=0)
        else:
            fnirs_filtered = fnirs_data
        
        return eeg_filtered, fnirs_filtered, motion_data
    
    def extract_features(self, eeg_data, fnirs_data):
        """Extract features from preprocessed signals."""
        features = []
        
        # EEG features
        for ch in range(eeg_data.shape[1]):
            ch_data = eeg_data[:, ch]
            
            # Time domain features
            features.append(np.mean(ch_data))
            features.append(np.std(ch_data))
            features.append(np.max(np.abs(ch_data)))
            features.append(np.mean(np.abs(ch_data)))
            features.append(np.sum(ch_data ** 2))
            
            # Frequency domain features
            freqs, psd = signal.welch(ch_data, fs=256, nperseg=min(256, len(ch_data)))
            
            # Band powers
            delta_idx = np.where((freqs >= 0.5) & (freqs <= 4))[0]
            theta_idx = np.where((freqs > 4) & (freqs <= 8))[0]
            alpha_idx = np.where((freqs > 8) & (freqs <= 13))[0]
            beta_idx = np.where((freqs > 13) & (freqs <= 30))[0]
            gamma_idx = np.where((freqs > 30) & (freqs <= 40))[0]
            
            features.append(np.sum(psd[delta_idx]) if len(delta_idx) > 0 else 0)
            features.append(np.sum(psd[theta_idx]) if len(theta_idx) > 0 else 0)
            features.append(np.sum(psd[alpha_idx]) if len(alpha_idx) > 0 else 0)
            features.append(np.sum(psd[beta_idx]) if len(beta_idx) > 0 else 0)
            features.append(np.sum(psd[gamma_idx]) if len(gamma_idx) > 0 else 0)
            
            # Peak frequency
            if len(psd) > 0:
                features.append(freqs[np.argmax(psd)])
            else:
                features.append(0)
            
            # Spectral entropy (simplified)
            if np.sum(psd) > 0:
                psd_norm = psd / np.sum(psd)
                entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                features.append(entropy)
            else:
                features.append(0)
        
        # fNIRS features
        for ch in range(fnirs_data.shape[1]):
            ch_data = fnirs_data[:, ch]
            
            features.append(np.mean(ch_data))
            features.append(np.std(ch_data))
            features.append(np.max(ch_data) - np.min(ch_data))
            
            # Linear trend
            if len(ch_data) > 1:
                x = np.arange(len(ch_data))
                slope, _ = np.polyfit(x, ch_data, 1)
                features.append(slope)
            else:
                features.append(0)
            
            # Peak timing
            if len(ch_data) > 0:
                features.append(np.argmax(ch_data) / len(ch_data))
            else:
                features.append(0.5)
        
        # Connectivity features
        if eeg_data.shape[1] >= 4:
            # Frontal alpha asymmetry (AF7 - AF8)
            alpha_af7 = features[1*12 + 7]  # AF7 alpha power
            alpha_af8 = features[2*12 + 7]  # AF8 alpha power
            features.append(alpha_af7 - alpha_af8)
            
            # Inter-channel coherence (simplified - using correlation as proxy)
            channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for i, j in channel_pairs:
                if eeg_data.shape[0] > 1:
                    corr = np.corrcoef(eeg_data[:, i], eeg_data[:, j])[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
                else:
                    features.append(0)
        else:
            features.extend([0] * 7)  # Add zeros if not enough channels
        
        return np.array(features)
    
    def predict(self, eeg_data, fnirs_data, motion_data, word_features):
        """Make prediction on current data window."""
        if self.model is None:
            return None, None
        
        try:
            # Preprocess signals
            eeg_proc, fnirs_proc, motion_proc = self.preprocess_signals(
                eeg_data, fnirs_data, motion_data
            )
            
            # Extract signal features
            signal_features = self.extract_features(eeg_proc, fnirs_proc)
            
            # Combine with word features
            all_features = np.concatenate([signal_features, word_features])
            
            # Standardize features
            all_features = self.scaler.transform(all_features.reshape(1, -1))
            
            # Convert to tensors
            eeg_tensor = torch.FloatTensor(eeg_proc.T).unsqueeze(0).to(self.device)
            fnirs_tensor = torch.FloatTensor(fnirs_proc.T).unsqueeze(0).to(self.device)
            motion_tensor = torch.FloatTensor(motion_proc.T).unsqueeze(0).to(self.device)
            features_tensor = torch.FloatTensor(all_features).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions, attention_weights = self.model(
                    eeg_tensor, fnirs_tensor, motion_tensor, features_tensor
                )
                probabilities = torch.softmax(predictions, dim=1)
            
            return probabilities.cpu().numpy(), attention_weights.cpu().numpy()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None


class LiveConfusionDetector:
    """Main application class for real-time confusion detection."""
    
    def __init__(self, root, port=8052):
        self.root = root
        self.root.title("Live Confusion Detection System")
        self.root.geometry("1600x900")
        
        # Network settings
        self.port = port
        self.socket = None
        
        # Initialize components
        self.model = None
        self.model_loaded = False
        
        # Data buffers
        self.buffer_size = 768  # 3 seconds at 256 Hz
        self.timestamps = deque(maxlen=self.buffer_size)
        self.eeg_channels = {
            'TP9': deque(maxlen=self.buffer_size),
            'AF7': deque(maxlen=self.buffer_size),
            'AF8': deque(maxlen=self.buffer_size),
            'TP10': deque(maxlen=self.buffer_size)
        }
        self.fnirs_channels = {
            f'Ch{i}_norm': deque(maxlen=self.buffer_size) for i in range(1, 5)
        }
        for i in range(1, 5):
            self.fnirs_channels[f'Ch{i}_raw'] = deque(maxlen=self.buffer_size)
        
        self.motion_channels = {
            'acc_x': deque(maxlen=self.buffer_size),
            'acc_y': deque(maxlen=self.buffer_size),
            'acc_z': deque(maxlen=self.buffer_size),
            'gyro_x': deque(maxlen=self.buffer_size),
            'gyro_y': deque(maxlen=self.buffer_size),
            'gyro_z': deque(maxlen=self.buffer_size)
        }
        
        # Reference channels
        self.ref_channels = {
            'DRL': deque(maxlen=self.buffer_size),
            'REF': deque(maxlen=self.buffer_size)
        }
        
        # Last values for interpolation
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Packet counts
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Tracking variables
        self.current_word = ""
        self.word_history = deque(maxlen=3)
        self.word_complexities = deque(maxlen=3)
        self.predicted_confusing_words = set()
        self.word_predictions = {}  # word -> confusion probability
        
        # Mode flags
        self.highlighting_mode = False
        self.dummy_highlighting_mode = False
        self.data_streaming = True
        self.data_flow_active = False
        
        # Text and highlighting
        self.text_content = ""
        self.word_positions = []  # List of (start, end, word) tuples
        self.manual_highlights = {'words': set(), 'sentences': set()}
        
        # Threading
        self.lock = threading.Lock()
        self.running = True
        
        # Setup GUI
        self.setup_gui()
        
        # Start data collection thread
        self.start_data_collection()
        
        # Start prediction loop
        self.start_prediction_loop()
    
    def parse_osc_message(self, data):
        """Parse OSC message from binary data (from eegtrainer.py)"""
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
        """Process incoming OSC message (from eegtrainer.py)"""
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
                    self.data_flow_active = True
                    
                    if self.eeg_packet_count <= 5:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                elif data_type == 'optics' and len(args) == 8:
                    for i, ch in enumerate([f'Ch{j}_{t}' for j in range(1,5) for t in ['norm', 'raw']]):
                        self.fnirs_channels[ch].append(args[i])
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                    
                    if self.fnirs_packet_count <= 5:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}, raw={args[4:]}")
                
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
                
                elif data_type == 'drlref' and len(args) >= 2:
                    self.ref_channels['DRL'].append(args[0])
                    self.ref_channels['REF'].append(args[1])
                    self.last_ref_data = args[:2]
    
    def setup_gui(self):
        """Create the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Text display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", 
                                        command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_text_btn = ttk.Button(control_frame, text="Load Text", 
                                       command=self.load_text)
        self.load_text_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.highlight_mode_btn = ttk.Button(control_frame, text="Toggle Highlighting", 
                                           command=self.toggle_highlighting)
        self.highlight_mode_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Text display
        self.text_display = tk.Text(left_frame, wrap=tk.WORD, font=("Arial", 14),
                                   width=60, height=25)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for highlighting
        self.text_display.tag_config("word_confusion", background="yellow")
        self.text_display.tag_config("sentence_confusion", background="orange")
        self.text_display.tag_config("predicted_confusion", background="lightblue")
        self.text_display.tag_config("manual_word", background="lightgreen")
        self.text_display.tag_config("manual_sentence", background="lightcoral")
        
        # Bind mouse events
        self.text_display.bind("<Motion>", self.track_cursor)
        self.text_display.bind("<Button-1>", self.on_left_click)
        self.text_display.bind("<B1-Motion>", self.on_drag)
        self.text_display.bind("<ButtonRelease-1>", self.on_left_release)
        self.text_display.bind("<Button-3>", self.on_right_click)
        
        # Right panel - Debug info and plots
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Debug info panel
        debug_frame = ttk.LabelFrame(right_frame, text="Debug Information")
        debug_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current word
        self.current_word_label = ttk.Label(debug_frame, text="1. Current Word: None", 
                                          font=("Arial", 10))
        self.current_word_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # EEG/fNIRS/Motion values
        self.signal_values_label = ttk.Label(debug_frame, 
                                           text="2. Signals: No data", 
                                           font=("Arial", 10))
        self.signal_values_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Word history
        self.word_history_label = ttk.Label(debug_frame, 
                                          text="3. Last 3 words: None", 
                                          font=("Arial", 10))
        self.word_history_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Model status
        self.model_status_label = ttk.Label(debug_frame, 
                                          text="4. Model: Not loaded", 
                                          font=("Arial", 10), foreground="red")
        self.model_status_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Data flow status with packet count
        self.data_flow_label = ttk.Label(debug_frame, 
                                        text="5. Data Flow: Inactive", 
                                        font=("Arial", 10), foreground="red")
        self.data_flow_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Predicted words
        self.predicted_words_text = tk.Text(debug_frame, height=4, width=50, 
                                          font=("Arial", 9))
        self.predicted_words_text.pack(padx=10, pady=5)
        self.predicted_words_text.insert(1.0, "6. Predicted confusing words:\nNone")
        self.predicted_words_text.config(state=tk.DISABLED)
        
        # Dummy highlighting button
        self.dummy_highlight_btn = ttk.Button(debug_frame, 
                                            text="7. Toggle Dummy Highlighting", 
                                            command=self.toggle_dummy_highlighting)
        self.dummy_highlight_btn.pack(pady=5)
        
        # Signal plots
        plot_frame = ttk.LabelFrame(right_frame, text="Signal Visualization")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True)
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.lines = {}
        self.axes[0].set_ylabel('EEG (µV)')
        self.axes[1].set_ylabel('fNIRS')
        self.axes[2].set_ylabel('Motion')
        self.axes[2].set_xlabel('Time (s)')
        
        for ax in self.axes:
            ax.grid(True, alpha=0.3)
        
        # Start plot updates
        self.update_plots()
    
    def load_model(self):
        """Load a trained model from file."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        
        if filename:
            self.model = ConfusionDetectorModel(filename)
            self.model_loaded = True
            self.model_status_label.config(text="4. Model: Loaded ✓", 
                                         foreground="green")
    
    def load_text(self):
        """Load text content from file."""
        filename = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                self.text_content = f.read()
            
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, self.text_content)
            
            # Parse word positions
            self.parse_word_positions()
    
    def parse_word_positions(self):
        """Parse and store word positions in the text."""
        self.word_positions = []
        text = self.text_display.get(1.0, tk.END)
        
        current_pos = 0
        for line_num, line in enumerate(text.split('\n')):
            word_start = None
            for i, char in enumerate(line):
                if char.isalnum() or char in "-'":
                    if word_start is None:
                        word_start = current_pos + i
                else:
                    if word_start is not None:
                        word_end = current_pos + i
                        word = text[word_start:word_end]
                        self.word_positions.append((word_start, word_end, word))
                        word_start = None
            
            # Handle word at end of line
            if word_start is not None:
                word_end = current_pos + len(line)
                word = text[word_start:word_end]
                self.word_positions.append((word_start, word_end, word))
            
            current_pos += len(line) + 1  # +1 for newline
    
    def track_cursor(self, event):
        """Track cursor position and identify current word."""
        if not self.data_streaming:
            return
        
        # Get cursor position in text
        index = self.text_display.index(f"@{event.x},{event.y}")
        position = self.text_display.count("1.0", index, "chars")[0]
        
        # Find word at position
        for start, end, word in self.word_positions:
            if start <= position < end:
                if word != self.current_word:
                    self.current_word = word
                    self.word_history.append(word)
                    complexity = extract_word_complexity_features(word)
                    self.word_complexities.append(complexity)
                    self.update_debug_info()
                break
    
    def update_debug_info(self):
        """Update debug information display."""
        # 1. Current word
        self.current_word_label.config(text=f"1. Current Word: {self.current_word}")
        
        # 2. Signal values (latest)
        with self.lock:
            if len(self.eeg_channels['TP9']) > 0:
                eeg_str = f"EEG: {self.eeg_channels['TP9'][-1]:.1f}µV"
                fnirs_str = f"fNIRS: {self.fnirs_channels['Ch1_norm'][-1]:.3f}" if len(self.fnirs_channels['Ch1_norm']) > 0 else "fNIRS: --"
                motion_str = f"Motion: {self.motion_channels['acc_x'][-1]:.2f}g" if len(self.motion_channels['acc_x']) > 0 else "Motion: --"
                signal_text = f"2. {eeg_str}, {fnirs_str}, {motion_str}"
            else:
                signal_text = "2. Signals: No data"
        
        self.signal_values_label.config(text=signal_text)
        
        # 3. Word history
        history_text = "3. Last 3 words: "
        if self.word_history:
            words_with_complexity = []
            for i, word in enumerate(self.word_history):
                if i < len(self.word_complexities):
                    complexity = self.word_complexities[i]
                    grade_level = complexity[13] if len(complexity) > 13 else 0
                    words_with_complexity.append(f"{word} (GL:{grade_level:.1f})")
            history_text += ", ".join(words_with_complexity)
        else:
            history_text += "None"
        
        self.word_history_label.config(text=history_text)
        
        # 5. Data flow with packet counts
        if self.data_flow_active:
            flow_text = f"5. Data Flow: Active ✓ (EEG: {self.eeg_packet_count}, fNIRS: {self.fnirs_packet_count})"
            color = "green"
        else:
            flow_text = "5. Data Flow: Inactive (No packets received)"
            color = "red"
        self.data_flow_label.config(text=flow_text, foreground=color)
        
        # 6. Predicted words
        self.predicted_words_text.config(state=tk.NORMAL)
        self.predicted_words_text.delete(1.0, tk.END)
        
        if self.predicted_confusing_words:
            # Sort by confidence
            sorted_words = sorted(self.word_predictions.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            text = "6. Predicted confusing words:\n"
            for word, prob in sorted_words:
                if prob > 0.3:  # Only show significant predictions
                    text += f"  {word}: {prob:.2%}\n"
        else:
            text = "6. Predicted confusing words:\nNone"
        
        self.predicted_words_text.insert(1.0, text)
        self.predicted_words_text.config(state=tk.DISABLED)
    
    def toggle_highlighting(self):
        """Toggle highlighting mode."""
        self.highlighting_mode = not self.highlighting_mode
        
        if self.highlighting_mode:
            self.highlight_mode_btn.config(text="Highlighting: ON")
            self.apply_predictions()
        else:
            self.highlight_mode_btn.config(text="Highlighting: OFF")
            self.clear_highlights()
    
    def toggle_dummy_highlighting(self):
        """Toggle dummy highlighting mode."""
        self.dummy_highlighting_mode = not self.dummy_highlighting_mode
        
        if self.dummy_highlighting_mode:
            self.dummy_highlight_btn.config(text="Dummy Mode: ON")
            self.data_streaming = False
            # Show predicted highlights
            self.apply_predictions()
        else:
            self.dummy_highlight_btn.config(text="Dummy Mode: OFF")
            self.data_streaming = True
            self.clear_manual_highlights()
    
    def apply_predictions(self):
        """Apply predicted confusion highlighting."""
        if not self.highlighting_mode and not self.dummy_highlighting_mode:
            return
        
        # Clear previous predicted highlights
        self.text_display.tag_remove("predicted_confusion", "1.0", tk.END)
        
        # Apply new highlights
        for word, prob in self.word_predictions.items():
            if prob > 0.5:  # Threshold for highlighting
                for start, end, w in self.word_positions:
                    if w == word:
                        start_idx = self.text_display.index(f"1.0 + {start} chars")
                        end_idx = self.text_display.index(f"1.0 + {end} chars")
                        self.text_display.tag_add("predicted_confusion", 
                                                start_idx, end_idx)
    
    def clear_highlights(self):
        """Clear all highlights."""
        self.text_display.tag_remove("predicted_confusion", "1.0", tk.END)
        self.text_display.tag_remove("word_confusion", "1.0", tk.END)
        self.text_display.tag_remove("sentence_confusion", "1.0", tk.END)
    
    def clear_manual_highlights(self):
        """Clear manual highlights."""
        self.text_display.tag_remove("manual_word", "1.0", tk.END)
        self.text_display.tag_remove("manual_sentence", "1.0", tk.END)
        self.manual_highlights = {'words': set(), 'sentences': set()}
    
    def on_left_click(self, event):
        """Handle left click for word selection."""
        if self.dummy_highlighting_mode:
            self.selection_start = self.text_display.index(f"@{event.x},{event.y}")
    
    def on_drag(self, event):
        """Handle drag for word selection."""
        if self.dummy_highlighting_mode:
            self.selection_end = self.text_display.index(f"@{event.x},{event.y}")
    
    def on_left_release(self, event):
        """Handle left release for word highlighting."""
        if self.dummy_highlighting_mode and hasattr(self, 'selection_start'):
            try:
                selected_text = self.text_display.get(self.selection_start, 
                                                    self.selection_end).strip()
                if selected_text:
                    self.manual_highlights['words'].add(selected_text)
                    self.text_display.tag_add("manual_word", 
                                            self.selection_start, 
                                            self.selection_end)
            except:
                pass
    
    def on_right_click(self, event):
        """Handle right click for sentence highlighting."""
        if self.dummy_highlighting_mode:
            # Find sentence boundaries
            index = self.text_display.index(f"@{event.x},{event.y}")
            text = self.text_display.get("1.0", tk.END)
            
            # Find sentence containing click position
            position = self.text_display.count("1.0", index, "chars")[0]
            
            # Find sentence start
            sentence_start = position
            while sentence_start > 0 and text[sentence_start-1] not in '.!?':
                sentence_start -= 1
            
            # Find sentence end
            sentence_end = position
            while sentence_end < len(text) and text[sentence_end] not in '.!?':
                sentence_end += 1
            
            if text[sentence_end] in '.!?':
                sentence_end += 1
            
            # Highlight sentence
            start_idx = self.text_display.index(f"1.0 + {sentence_start} chars")
            end_idx = self.text_display.index(f"1.0 + {sentence_end} chars")
            
            sentence = text[sentence_start:sentence_end].strip()
            if sentence:
                self.manual_highlights['sentences'].add(sentence)
                self.text_display.tag_add("manual_sentence", start_idx, end_idx)
    
    def start_data_collection(self):
        """Start OSC data collection thread."""
        self.osc_thread = threading.Thread(target=self.osc_receiver, daemon=True)
        self.osc_thread.start()
    
    def osc_receiver(self):
        """Receive and parse OSC messages using eegtrainer.py's method."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Bind to 0.0.0.0 instead of localhost to receive from network
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            self.running = True
            
            print(f"📡 Listening for OSC data on UDP port {self.port}")
            print("Make sure Muse Direct is streaming to this computer's IP address")
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(4096)  # Increased buffer size
                    self.packet_count += 1
                    message = self.parse_osc_message(data)
                    if message and self.data_streaming:
                        self.process_osc_message(message)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Receiver error: {e}")
        
        except Exception as e:
            print(f"❌ Failed to start receiver: {e}")
        
        finally:
            if self.socket:
                self.socket.close()
    
    def start_prediction_loop(self):
        """Start the prediction loop."""
        self.prediction_thread = threading.Thread(target=self.prediction_loop, 
                                                daemon=True)
        self.prediction_thread.start()
    
    def prediction_loop(self):
        """Run predictions on current data windows."""
        while self.running:
            if self.model_loaded and self.data_flow_active and len(self.timestamps) >= self.buffer_size:
                try:
                    # Get current data window
                    with self.lock:
                        eeg_data = np.array([
                            list(self.eeg_channels['TP9']),
                            list(self.eeg_channels['AF7']),
                            list(self.eeg_channels['AF8']),
                            list(self.eeg_channels['TP10'])
                        ]).T
                        
                        fnirs_data = np.array([
                            list(self.fnirs_channels[f'Ch{i}_norm']) for i in range(1, 5)
                        ] + [
                            list(self.fnirs_channels[f'Ch{i}_raw']) for i in range(1, 5)
                        ]).T
                        
                        motion_data = np.array([
                            list(self.motion_channels['acc_x']),
                            list(self.motion_channels['acc_y']),
                            list(self.motion_channels['acc_z']),
                            list(self.motion_channels['gyro_x']),
                            list(self.motion_channels['gyro_y']),
                            list(self.motion_channels['gyro_z'])
                        ]).T
                    
                    # Ensure correct shapes
                    if eeg_data.shape[0] < self.buffer_size:
                        continue
                    
                    # Get word features for current word
                    word_features = extract_word_complexity_features(self.current_word)
                    
                    # Make prediction
                    probs, attention = self.model.predict(
                        eeg_data[-self.buffer_size:],
                        fnirs_data[-self.buffer_size:] if fnirs_data.shape[0] >= self.buffer_size else np.zeros((self.buffer_size, 8)),
                        motion_data[-self.buffer_size:] if motion_data.shape[0] >= self.buffer_size else np.zeros((self.buffer_size, 6)),
                        word_features
                    )
                    
                    if probs is not None:
                        # Update predictions
                        confusion_prob = probs[0][1] + probs[0][2]  # Word + Sentence confusion
                        
                        if self.current_word and confusion_prob > 0.3:
                            self.word_predictions[self.current_word] = confusion_prob
                            self.predicted_confusing_words.add(self.current_word)
                        
                        # Update display if highlighting is on
                        if self.highlighting_mode:
                            self.root.after(0, self.apply_predictions)
                        
                        # Update debug info
                        self.root.after(0, self.update_debug_info)
                
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            time.sleep(0.1)  # 10 Hz prediction rate
    
    def update_plots(self):
        """Update signal visualization plots."""
        if not self.running:
            return
        
        with self.lock:
            # Update EEG plot
            if len(self.eeg_channels['TP9']) > 10:
                time_vec = np.linspace(-3, 0, len(self.eeg_channels['TP9']))
                
                self.axes[0].clear()
                for i, (name, data) in enumerate(self.eeg_channels.items()):
                    if len(data) > 10:
                        # Apply high-pass filter for visualization
                        b, a = signal.butter(2, 1, btype='high', fs=256)
                        filtered = signal.filtfilt(b, a, list(data))
                        self.axes[0].plot(time_vec, filtered - i*50, 
                                        label=name, alpha=0.8)
                
                self.axes[0].set_ylabel('EEG (µV)')
                self.axes[0].legend(loc='upper right', fontsize=8)
                self.axes[0].set_ylim(-200, 50)
                self.axes[0].grid(True, alpha=0.3)
            
            # Update fNIRS plot
            if len(self.fnirs_channels['Ch1_norm']) > 10:
                time_vec = np.linspace(-3, 0, len(self.fnirs_channels['Ch1_norm']))
                
                self.axes[1].clear()
                for i in range(1, 5):
                    ch_data = list(self.fnirs_channels[f'Ch{i}_norm'])
                    if len(ch_data) > 10:
                        self.axes[1].plot(time_vec, ch_data, 
                                        label=f'Ch{i}', alpha=0.8)
                
                self.axes[1].set_ylabel('fNIRS (norm)')
                self.axes[1].legend(loc='upper right', fontsize=8)
                self.axes[1].grid(True, alpha=0.3)
            
            # Update motion plot
            if len(self.motion_channels['acc_x']) > 10:
                time_vec = np.linspace(-3, 0, len(self.motion_channels['acc_x']))
                
                self.axes[2].clear()
                for name, data in self.motion_channels.items():
                    if len(data) > 10 and 'acc' in name:
                        self.axes[2].plot(time_vec, list(data), 
                                        label=name, alpha=0.8)
                
                self.axes[2].set_ylabel('Acceleration (g)')
                self.axes[2].set_xlabel('Time (s)')
                self.axes[2].legend(loc='upper right', fontsize=8)
                self.axes[2].grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Schedule next update
        self.root.after(100, self.update_plots)  # 10 Hz update
    
    def on_closing(self):
        """Clean up when closing the application."""
        self.running = False
        if self.socket:
            self.socket.close()
        self.root.quit()
        self.root.destroy()


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("   LIVE CONFUSION DETECTION SYSTEM")
    print("="*60)
    print("\n📡 Starting OSC receiver...")
    print("Make sure Muse Direct is streaming to this computer's IP address on port 8052")
    print("\n" + "="*60 + "\n")
    
    root = tk.Tk()
    app = LiveConfusionDetector(root, port=8052)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()