#!/usr/bin/env python3
"""
Real-Time Word-Level Confusion Detection System
Uses trained model to detect confusion while reading academic text
"""

import tkinter as tk
from tkinter import ttk, font
import numpy as np
import socket
import struct
import threading
import time
from collections import deque, defaultdict
from scipy import signal, stats
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Sample academic texts for testing
ACADEMIC_TEXTS = [
"""The hippocampal-entorhinal complex supports episodic memory through the formation of cognitive maps. Place cells in the hippocampus exhibit spatially selective firing patterns, while grid cells in the medial entorhinal cortex demonstrate hexagonal firing fields. This spatial representation system provides a neural substrate for navigation and memory encoding. Recent evidence suggests that these cells also encode non-spatial dimensions, supporting abstract cognitive mapping beyond physical space.""",

"""Quantum entanglement represents a fundamental phenomenon where particles exhibit correlated properties regardless of spatial separation. Bell's inequality violations demonstrate that local hidden variable theories cannot explain quantum correlations. The EPR paradox highlighted tensions between quantum mechanics and classical intuitions about locality and realism. Modern quantum information theory leverages entanglement for quantum computing and cryptographic protocols.""",

"""The extracellular tissue medium was approximated as an isotropic ohmic conductor. A disk electrode acting as a current source with ground located at infinity was placed below (~10 µm) the modeled RGC. Since the extracellular domain was isotropic, the extracellular voltage at each point in space was calculated using Laplace's equation. The activation threshold was defined as the lowest current amplitude required to elicit a spike."""
]

class RealtimeConfusionDetector:
    def __init__(self, model_path=None, port=8052):
        # Window settings
        self.root = tk.Tk()
        self.root.title("Real-Time Confusion Detection")
        self.root.geometry("1400x900")
        
        # Network settings
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers (matching the analysis file)
        self.buffer_size = 1024  # ~4 seconds at 256Hz
        self.sample_rate = 256
        
        # Channel data
        self.timestamps = deque(maxlen=self.buffer_size)
        self.eeg_data = {
            'TP9': deque(maxlen=self.buffer_size),
            'AF7': deque(maxlen=self.buffer_size),
            'AF8': deque(maxlen=self.buffer_size),
            'TP10': deque(maxlen=self.buffer_size)
        }
        self.fnirs_data = {
            'Ch1_norm': deque(maxlen=self.buffer_size),
            'Ch2_norm': deque(maxlen=self.buffer_size),
            'Ch3_norm': deque(maxlen=self.buffer_size),
            'Ch4_norm': deque(maxlen=self.buffer_size)
        }
        
        # Preprocessing filters
        self.setup_filters()
        
        # Model and prediction
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor(sample_rate=self.sample_rate)
        
        # Current state
        self.current_text_index = 0
        self.current_word_index = 0
        self.current_word = ""
        self.words = []
        self.word_positions = []  # Store word positions in text widget
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Last 5 predictions
        self.word_predictions = {}  # word_index -> prediction
        
        # Confusion tracking
        self.confusion_counts = defaultdict(int)
        self.total_confusion_score = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # GUI elements
        self.setup_gui()
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Using random predictions for demo.")
        
        # Start network receiver
        self.start_receiver()
        
    def setup_filters(self):
        """Setup signal processing filters"""
        nyquist = self.sample_rate / 2
        
        # Bandpass filter (0.5-50 Hz)
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        if low < 1 and high < 1:
            self.b_band, self.a_band = signal.butter(4, [low, high], btype='band')
        
        # Notch filters
        self.notch_filters = []
        for freq in [60, 120]:
            if freq < nyquist:
                notch_freq = freq / nyquist
                b_notch, a_notch = signal.iirnotch(notch_freq, Q=30)
                self.notch_filters.append((b_notch, a_notch))
        
        # fNIRS lowpass filter (0.5 Hz)
        if 0.5 < nyquist:
            fnirs_low = 0.5 / nyquist
            self.b_fnirs, self.a_fnirs = signal.butter(4, fnirs_low, btype='low')
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Text display
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Text controls
        text_control_frame = ttk.Frame(left_frame)
        text_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(text_control_frame, text="Previous Text", 
                   command=self.previous_text).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(text_control_frame, text="Next Text", 
                   command=self.next_text).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(text_control_frame, text="Reset Highlights", 
                   command=self.reset_highlights).grid(row=0, column=2)
        
        # Text display with custom font
        text_font = font.Font(family="Georgia", size=14)
        self.text_widget = tk.Text(left_frame, wrap=tk.WORD, font=text_font,
                                   width=60, height=20, padx=20, pady=20,
                                   bg='#f9f9f9', cursor="arrow")
        self.text_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags
        self.text_widget.tag_configure("confused_word", background="#ffcccc", foreground="#cc0000")
        self.text_widget.tag_configure("confused_sentence", background="#ffe6cc", foreground="#cc6600")
        self.text_widget.tag_configure("not_confused", background="#ccffcc", foreground="#006600")
        self.text_widget.tag_configure("current_word", underline=True, font=font.Font(family="Georgia", size=14, weight="bold"))
        
        # Bind mouse motion
        self.text_widget.bind("<Motion>", self.on_mouse_motion)
        
        # Right panel - Stats and visualizations
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Stats frame
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Confusion percentage
        self.confusion_label = ttk.Label(stats_frame, text="Overall Confusion: 0.0%", 
                                        font=font.Font(size=12, weight="bold"))
        self.confusion_label.grid(row=0, column=0, sticky=tk.W)
        
        # Current word
        self.word_label = ttk.Label(stats_frame, text="Current Word: -", 
                                   font=font.Font(size=11))
        self.word_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Prediction
        self.prediction_label = ttk.Label(stats_frame, text="Prediction: -", 
                                         font=font.Font(size=11))
        self.prediction_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Data stream status
        data_frame = ttk.LabelFrame(right_frame, text="Data Stream", padding="10")
        data_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.eeg_status = ttk.Label(data_frame, text="EEG: Waiting...", font=font.Font(size=9))
        self.eeg_status.grid(row=0, column=0, sticky=tk.W)
        
        self.fnirs_status = ttk.Label(data_frame, text="fNIRS: Waiting...", font=font.Font(size=9))
        self.fnirs_status.grid(row=1, column=0, sticky=tk.W)
        
        # PCA visualization
        self.setup_pca_plot(right_frame)
        
        # Feature importance display
        feature_frame = ttk.LabelFrame(right_frame, text="Top Active Features", padding="10")
        feature_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.feature_labels = []
        for i in range(5):
            label = ttk.Label(feature_frame, text=f"{i+1}. -", font=font.Font(size=9))
            label.grid(row=i, column=0, sticky=tk.W)
            self.feature_labels.append(label)
        
        # Load initial text
        self.load_text(self.current_text_index)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
    
    def setup_pca_plot(self, parent):
        """Setup PCA visualization plot"""
        pca_frame = ttk.LabelFrame(parent, text="Feature Space (PCA)", padding="10")
        pca_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create matplotlib figure
        self.pca_fig = Figure(figsize=(4, 3), dpi=100)
        self.pca_ax = self.pca_fig.add_subplot(111)
        self.pca_ax.set_xlabel('PC1')
        self.pca_ax.set_ylabel('PC2')
        self.pca_ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.pca_canvas = FigureCanvasTkAgg(self.pca_fig, master=pca_frame)
        self.pca_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty scatter plots
        self.pca_scatter_baseline = self.pca_ax.scatter([], [], c='green', alpha=0.5, label='Baseline')
        self.pca_scatter_confused = self.pca_ax.scatter([], [], c='red', alpha=0.5, label='Confused')
        self.pca_scatter_current = self.pca_ax.scatter([], [], c='blue', s=100, marker='*', label='Current')
        self.pca_ax.legend()
        
        self.pca_fig.tight_layout()
    
    def load_text(self, index):
        """Load a text passage"""
        self.current_text_index = index % len(ACADEMIC_TEXTS)
        text = ACADEMIC_TEXTS[self.current_text_index]
        
        # Clear and insert text
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.insert('1.0', text)
        
        # Parse words and positions
        self.words = []
        self.word_positions = []
        
        # Get all word positions
        current_pos = '1.0'
        while True:
            # Find next word boundary
            word_start = self.text_widget.search(r'\S', current_pos, tk.END, regexp=True)
            if not word_start:
                break
            
            word_end = self.text_widget.search(r'\s', word_start, tk.END, regexp=True)
            if not word_end:
                word_end = tk.END
            
            word = self.text_widget.get(word_start, word_end)
            if word:
                self.words.append(word)
                self.word_positions.append((word_start, word_end))
            
            current_pos = word_end
        
        # Reset predictions
        self.word_predictions = {}
        self.reset_highlights()
    
    def on_mouse_motion(self, event):
        """Handle mouse motion over text"""
        # Get mouse position in text widget
        mouse_pos = self.text_widget.index(f"@{event.x},{event.y}")
        
        # Find which word the mouse is over
        for i, (start, end) in enumerate(self.word_positions):
            if self.text_widget.compare(mouse_pos, ">=", start) and \
               self.text_widget.compare(mouse_pos, "<", end):
                self.current_word_index = i
                self.current_word = self.words[i]
                self.update_current_word_display()
                break
    
    def update_current_word_display(self):
        """Update the current word highlighting and label"""
        # Remove previous highlight
        self.text_widget.tag_remove("current_word", '1.0', tk.END)
        
        # Add current word highlight
        if 0 <= self.current_word_index < len(self.word_positions):
            start, end = self.word_positions[self.current_word_index]
            self.text_widget.tag_add("current_word", start, end)
            
        # Update label
        self.word_label.config(text=f"Current Word: {self.current_word}")
    
    def process_data_packet(self, data_type, values, timestamp):
        """Process incoming data packet"""
        with self.lock:
            self.timestamps.append(timestamp)
            
            if data_type == 'eeg':
                channels = ['TP9', 'AF7', 'AF8', 'TP10']
                for ch, val in zip(channels, values):
                    self.eeg_data[ch].append(val)
                
                # Update status
                self.root.after(0, lambda: self.eeg_status.config(
                    text=f"EEG: {values[0]:.1f}, {values[1]:.1f}, {values[2]:.1f}, {values[3]:.1f}"))
                
            elif data_type == 'fnirs':
                channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
                for ch, val in zip(channels, values[:4]):
                    self.fnirs_data[ch].append(val)
                
                # Update status
                self.root.after(0, lambda: self.fnirs_status.config(
                    text=f"fNIRS: {values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}, {values[3]:.2f}"))
        
        # Trigger prediction if we have enough data
        if len(self.timestamps) >= 512:  # 2 seconds of data
            self.predict_confusion()
    
    def extract_features(self):
        """Extract features from current buffer"""
        with self.lock:
            if len(self.timestamps) < 256:  # Need at least 1 second
                return None
            
            # Get recent data
            timestamps = np.array(self.timestamps)
            
            # EEG data
            eeg_array = np.column_stack([
                np.array(self.eeg_data['TP9']),
                np.array(self.eeg_data['AF7']),
                np.array(self.eeg_data['AF8']),
                np.array(self.eeg_data['TP10'])
            ])
            
            # fNIRS data
            fnirs_array = np.column_stack([
                np.array(self.fnirs_data['Ch1_norm']),
                np.array(self.fnirs_data['Ch2_norm']),
                np.array(self.fnirs_data['Ch3_norm']),
                np.array(self.fnirs_data['Ch4_norm'])
            ])
            
            # Use last 2 seconds
            window_size = min(512, len(timestamps))
            eeg_window = eeg_array[-window_size:]
            fnirs_window = fnirs_array[-window_size:]
            
            # Extract features using the same method as training
            features = self.feature_extractor.extract_all_features(
                eeg_window, fnirs_window, self.current_word
            )
            
            return features
    
    def predict_confusion(self):
        """Predict confusion for current state"""
        features = self.extract_features()
        if features is None:
            return
        
        # Predict
        if self.model is not None:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0]
        else:
            # Demo mode - random predictions
            prediction = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            proba = np.array([0.7, 0.2, 0.1])
        
        # Add to history
        self.prediction_history.append(prediction)
        
        # Smooth prediction
        if len(self.prediction_history) >= 3:
            smoothed = int(np.median(self.prediction_history))
        else:
            smoothed = prediction
        
        # Update word prediction
        self.word_predictions[self.current_word_index] = smoothed
        
        # Update display
        pred_text = ['Not Confused', 'Word Confusion', 'Sentence Confusion'][smoothed]
        conf_prob = proba[1] + proba[2]  # Total confusion probability
        
        self.root.after(0, lambda: self.update_display(pred_text, conf_prob, features))
    
    def update_display(self, prediction_text, confusion_prob, features):
        """Update GUI with prediction results"""
        # Update prediction label
        self.prediction_label.config(text=f"Prediction: {prediction_text} ({confusion_prob:.1%})")
        
        # Update overall confusion
        self.total_confusion_score = self.total_confusion_score * 0.95 + confusion_prob * 0.05
        self.confusion_label.config(text=f"Overall Confusion: {self.total_confusion_score:.1%}")
        
        # Update word highlighting
        if 0 <= self.current_word_index < len(self.word_positions):
            start, end = self.word_positions[self.current_word_index]
            
            # Remove old tags
            for tag in ['confused_word', 'confused_sentence', 'not_confused']:
                self.text_widget.tag_remove(tag, start, end)
            
            # Add new tag based on prediction
            prediction = self.word_predictions.get(self.current_word_index, 0)
            if prediction == 1:
                self.text_widget.tag_add('confused_word', start, end)
            elif prediction == 2:
                # Highlight period at end of sentence
                sentence_end = self.find_sentence_end(self.current_word_index)
                if sentence_end:
                    self.text_widget.tag_add('confused_sentence', sentence_end[0], sentence_end[1])
            else:
                self.text_widget.tag_add('not_confused', start, end)
        
        # Update feature importance display
        if hasattr(self, 'feature_names') and features is not None:
            top_indices = np.argsort(np.abs(features))[-5:][::-1]
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    self.feature_labels[i].config(
                        text=f"{i+1}. {self.feature_names[idx]}: {features[idx]:.2f}")
        
        # Update PCA plot (periodically)
        if hasattr(self, 'pca') and np.random.random() < 0.1:  # Update 10% of the time
            self.update_pca_plot(features)
    
    def find_sentence_end(self, word_index):
        """Find the period at the end of the current sentence"""
        # Search forward from current word
        for i in range(word_index, len(self.words)):
            if any(self.words[i].endswith(p) for p in ['.', '!', '?']):
                start, end = self.word_positions[i]
                # Find the punctuation position
                word = self.words[i]
                if word[-1] in '.!?':
                    punct_start = self.text_widget.index(f"{end} -1c")
                    return (punct_start, end)
        return None
    
    def update_pca_plot(self, current_features):
        """Update PCA visualization"""
        # This is a simplified version - in production you'd maintain a history
        # of features and their labels for better visualization
        pass
    
    def load_model(self, model_path):
        """Load the trained model and associated data"""
        try:
            # Load the model package
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            # Fit PCA on some dummy data for visualization
            n_features = len(self.feature_names)
            dummy_data = np.random.randn(100, n_features)
            self.pca = PCA(n_components=2)
            self.pca.fit(dummy_data)
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in demo mode with random predictions.")
    
    def reset_highlights(self):
        """Reset all highlighting"""
        for tag in ['confused_word', 'confused_sentence', 'not_confused']:
            self.text_widget.tag_remove(tag, '1.0', tk.END)
        self.word_predictions = {}
    
    def previous_text(self):
        """Load previous text"""
        self.load_text(self.current_text_index - 1)
    
    def next_text(self):
        """Load next text"""
        self.load_text(self.current_text_index + 1)
    
    def start_receiver(self):
        """Start the UDP receiver thread"""
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.settimeout(0.1)
        
        self.running = True
        self.receiver_thread = threading.Thread(target=self.receiver_loop)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        print(f"Listening for data on port {self.port}...")
    
    def receiver_loop(self):
        """Main receiver loop"""
        parser = OSCParser()
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = parser.parse(data)
                
                if message:
                    address = message['address']
                    args = message['args']
                    
                    # Extract data type
                    parts = address.strip('/').split('/')
                    if len(parts) >= 2:
                        data_type = parts[1]
                        
                        if data_type == 'eeg' and len(args) == 4:
                            self.process_data_packet('eeg', args, time.time())
                        elif data_type == 'optics' and len(args) >= 4:
                            self.process_data_packet('fnirs', args, time.time())
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def run(self):
        """Start the GUI main loop"""
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.socket:
            self.socket.close()

class OSCParser:
    """Simple OSC message parser"""
    def parse(self, data):
        try:
            offset = 0
            
            # Parse address
            address, offset = self.parse_string(data, offset)
            if not address:
                return None
            
            # Parse type tags
            type_tags, offset = self.parse_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            # Parse arguments
            args = []
            for tag in type_tags[1:]:
                if tag == 'f':  # float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
            
            return {'address': address, 'args': args}
        except:
            return None
    
    def parse_string(self, data, offset):
        """Parse null-terminated string"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        offset = ((end + 4) // 4) * 4
        return string, offset

class FeatureExtractor:
    """Feature extraction matching the training code"""
    def __init__(self, sample_rate=256):
        self.sample_rate = sample_rate
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
    def extract_all_features(self, eeg_window, fnirs_window, word):
        """Extract all features from data windows"""
        features = []
        
        # EEG features per channel
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
                
                # Band powers
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
                
                # Peak frequency and entropy
                peak_freq = freqs[np.argmax(psd)]
                features.append(peak_freq)
                
                psd_norm = psd / (np.sum(psd) + 1e-10)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                features.append(spectral_entropy)
            else:
                features.extend([0] * 7)
        
        # fNIRS features
        for ch in range(fnirs_window.shape[1]):
            channel_data = fnirs_window[:, ch]
            
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(channel_data) - np.min(channel_data),
            ])
            
            # Slope
            if len(channel_data) > 1:
                time_vector = np.arange(len(channel_data))
                slope, _ = np.polyfit(time_vector, channel_data, 1)
                features.append(slope)
            else:
                features.append(0)
            
            # Time to peak
            peak_idx = np.argmax(channel_data)
            time_to_peak = peak_idx / self.sample_rate
            features.append(time_to_peak)
        
        # Connectivity features
        if eeg_window.shape[1] == 4:
            # Frontal asymmetry
            frontal_alpha_left = self._get_band_power(eeg_window[:, 1], 'alpha')
            frontal_alpha_right = self._get_band_power(eeg_window[:, 2], 'alpha')
            frontal_asymmetry = (frontal_alpha_right - frontal_alpha_left) / \
                               (frontal_alpha_right + frontal_alpha_left + 1e-10)
            features.append(frontal_asymmetry)
            
            # Phase synchronization
            for i in range(eeg_window.shape[1]):
                for j in range(i+1, eeg_window.shape[1]):
                    sync = self._phase_sync(eeg_window[:, i], eeg_window[:, j])
                    features.append(sync)
        
        # Word features
        features.extend([
            len(word),
            self._count_syllables(word),
            1 if any(c.isdigit() for c in word) else 0,
            0,  # Previously confused (not tracked in real-time)
            0   # Confusion count (not tracked in real-time)
        ])
        
        return np.array(features)
    
    def _get_band_power(self, data, band):
        """Calculate power in specific frequency band"""
        if len(data) < 64:
            return 0
        
        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=min(len(data), 64))
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        low, high = bands[band]
        band_mask = (freqs >= low) & (freqs <= high)
        
        if np.any(band_mask):
            return np.mean(psd[band_mask])
        return 0
    
    def _phase_sync(self, signal1, signal2):
        """Calculate phase synchronization"""
        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
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

def main():
    """Main entry point"""
    import sys
    
    # Check for model path
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("="*60)
    print("REAL-TIME CONFUSION DETECTION SYSTEM")
    print("="*60)
    print()
    print("Instructions:")
    print("1. Start your Muse data stream")
    print("2. Move your mouse over words as you read")
    print("3. Words will be highlighted based on confusion level:")
    print("   - Green: Not confused")
    print("   - Red: Confused by word")
    print("   - Orange: Confused by sentence")
    print()
    
    # Create and run detector
    detector = RealtimeConfusionDetector(model_path=model_path)
    detector.run()

if __name__ == "__main__":
    main()