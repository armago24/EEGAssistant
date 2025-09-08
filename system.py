#!/usr/bin/env python3
"""
Live Word-Level Confusion Detection System
Trains on pre-collected data, then predicts confusion in real-time
Highlights potentially confusing words/ideas as you read
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import socket
import struct
import threading
import time
from collections import deque, defaultdict
import tkinter as tk
from tkinter import font as tkfont, filedialog
import warnings
import pickle
import os
from datetime import datetime
import colorsys

warnings.filterwarnings('ignore')

# Academic texts for testing
TEST_TEXTS = [
"""Quantum entanglement represents a correlation between quantum systems that persists regardless of spatial separation. When two particles become entangled through interaction, measurements performed on one particle instantaneously affect the quantum state of its partner, a phenomenon Einstein famously derided as "spooky action at a distance." This non-local correlation cannot transmit classical information faster than light, preserving relativistic causality while challenging classical intuitions about locality and realism. Bell's inequalities provide experimental tests distinguishing quantum mechanical predictions from local hidden variable theories.""",

"""The renormalization group approach reveals how physical systems behave across different length scales. Starting from microscopic degrees of freedom, successive coarse-graining transformations generate an effective description at larger scales. Fixed points of these transformations correspond to scale-invariant theories, while relevant, marginal, and irrelevant operators determine the system's macroscopic behavior. This framework unifies diverse phenomena from critical phase transitions to quantum field theory, demonstrating how apparently different systems can share identical long-distance physics through universality.""",

"""Topological insulators exhibit insulating behavior in their bulk while maintaining metallic surface states protected by time-reversal symmetry. These surface states possess linear dispersion relations forming Dirac cones in momentum space, with spin-momentum locking preventing backscattering from non-magnetic impurities. The topological invariant characterizing these materials arises from the Berry phase accumulated by electronic wavefunctions across the Brillouin zone. This paradigm extends to higher-order topological phases exhibiting protected corner or hinge states."""
]

class LiveConfusionDetector:
    def __init__(self):
        """Initialize the live detection system"""
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        
        # Network settings for Muse data
        self.port = 8052
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = 1000
        self.timestamps = deque(maxlen=self.buffer_size)
        self.eeg_buffer = deque(maxlen=self.buffer_size)
        self.fnirs_buffer = deque(maxlen=self.buffer_size)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # UI elements
        self.root = None
        self.text_display = None
        self.current_text_index = 0
        
        # Tracking
        self.current_word = ""
        self.current_word_index = ""
        self.word_predictions = {}  # word_index -> confusion_probability
        self.prediction_history = deque(maxlen=100)  # For smoothing
        
        # Model parameters
        self.sample_rate = 256
        self.window_size = 2.0  # seconds
        self.prediction_threshold = 0.5  # Threshold for highlighting
        
        # Colors for visualization
        self.confusion_gradient = self._generate_color_gradient()
        
    def _generate_color_gradient(self):
        """Generate color gradient from green to red for confusion levels"""
        gradient = []
        for i in range(101):  # 0-100% confusion
            # HSV: green (120°) to red (0°)
            hue = 120 * (1 - i/100) / 360
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            )
            gradient.append(hex_color)
        return gradient
    
    def load_training_data(self, filepath):
        """Load and process training data"""
        print(f"\n{'='*50}")
        print("LOADING TRAINING DATA")
        print(f"{'='*50}")
        
        data = np.load(filepath, allow_pickle=True)
        
        # Extract data arrays
        self.training_data = {
            'timestamps': data['timestamps'],
            'eeg': data['eeg'],
            'fnirs': data['fnirs'] if 'fnirs' in data else None,
            'event_timestamps': data['event_timestamps'],
            'event_types': data['event_types'],
            'event_words': data['event_words'] if 'event_words' in data else None,
            'words': data['words'] if 'words' in data else None,
            'metadata': data['metadata'].item() if 'metadata' in data else {}
        }
        
        print(f"  Duration: {(self.training_data['timestamps'][-1] - self.training_data['timestamps'][0]):.1f}s")
        print(f"  Samples: {len(self.training_data['timestamps'])}")
        print(f"  Events: {len(self.training_data['event_timestamps'])}")
        
        # Build feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.training_data['metadata'].get('sample_rate', 256),
            eeg_channels=self.training_data['metadata'].get('eeg_channels', ['TP9', 'AF7', 'AF8', 'TP10'])
        )
        
        return self.training_data
    
    def train_model(self):
        """Train the confusion detection model"""
        print(f"\n{'='*50}")
        print("TRAINING CONFUSION MODEL")
        print(f"{'='*50}")
        
        # Create dataset
        X, y, words, timestamps = self._create_training_dataset()
        
        if len(X) == 0:
            raise ValueError("No training data extracted!")
        
        # Split data (using all for training in live demo)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print("\nTraining XGBoost model...")
        
        # Binary classification: confused or not
        y_binary = (y > 0).astype(int)
        
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_scaled, y_binary)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = np.mean(y_pred == y_binary)
        
        print(f"\nTraining complete!")
        print(f"  Training accuracy: {accuracy:.3f}")
        print(f"  Confused words: {sum(y_binary)} / {len(y_binary)}")
        
        return self.model, self.scaler
    
    def _create_training_dataset(self):
        """Create training dataset from loaded data"""
        # Process confused words
        confused_words = defaultdict(list)
        
        if self.training_data['event_words'] is not None:
            for timestamp, event_type, word in zip(
                self.training_data['event_timestamps'],
                self.training_data['event_types'],
                self.training_data['event_words']
            ):
                if word and word.strip():
                    clean_word = word.strip().lower()
                    confused_words[clean_word].append((timestamp, event_type))
        
        # Extract features
        X = []
        y = []
        words_list = []
        timestamps_list = []
        
        # Process confused words
        for word, events in confused_words.items():
            for timestamp, event_type in events:
                features = self.feature_extractor.extract_features(
                    word, timestamp,
                    self.training_data['timestamps'],
                    self.training_data['eeg'],
                    self.training_data['fnirs']
                )
                if features is not None:
                    X.append(features)
                    y.append(1 if event_type == 'word_confusion' else 2)
                    words_list.append(word)
                    timestamps_list.append(timestamp)
        
        # Add baseline samples
        if self.training_data['words'] is not None:
            # Get confused timestamps
            confused_times = [t for events in confused_words.values() for t, _ in events]
            
            baseline_count = 0
            for timestamp, word in zip(self.training_data['timestamps'], self.training_data['words']):
                if word and word.strip() and len(word) >= 3:
                    # Skip if too close to confusion event
                    if any(abs(timestamp - ct) < 3.0 for ct in confused_times):
                        continue
                    
                    features = self.feature_extractor.extract_features(
                        word, timestamp,
                        self.training_data['timestamps'],
                        self.training_data['eeg'],
                        self.training_data['fnirs']
                    )
                    if features is not None:
                        X.append(features)
                        y.append(0)
                        words_list.append(word)
                        timestamps_list.append(timestamp)
                        baseline_count += 1
                        
                        if baseline_count >= len(confused_words) * 2:
                            break
        
        return np.array(X), np.array(y), words_list, timestamps_list
    
    def start_live_detection(self):
        """Start live confusion detection"""
        print(f"\n{'='*50}")
        print("STARTING LIVE DETECTION")
        print(f"{'='*50}")
        
        # Start UDP receiver for Muse data
        self.start_muse_receiver()
        
        # Create and show UI
        self.create_ui()
        
        # Start prediction loop
        self.start_prediction_loop()
        
        print("\n✅ Live detection started!")
        print("\n📖 Instructions:")
        print("  • Read the text naturally, moving your cursor along")
        print("  • Words will be highlighted based on predicted confusion")
        print("  • Red = High confusion, Green = Low confusion")
        print("  • Click words to provide feedback")
        print("  • Press ← → to change texts")
        
        # Run UI
        self.root.mainloop()
    
    def create_ui(self):
        """Create the live detection UI"""
        self.root = tk.Tk()
        self.root.title("🧠 Live Confusion Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        
        # Header
        header = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        header.pack_propagate(False)
        
        title = tk.Label(header,
                        text="LIVE CONFUSION DETECTION",
                        font=('Arial', 24, 'bold'),
                        fg='#FFD93D',
                        bg='#1a1a1a')
        title.pack(pady=10)
        
        instructions = tk.Label(header,
                               text="Move cursor along text • Red = Predicted confusion • Click for feedback",
                               font=('Arial', 14),
                               fg='#4ECDC4',
                               bg='#1a1a1a')
        instructions.pack()
        
        # Main text area
        text_frame = tk.Frame(self.root, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_display = tk.Text(text_frame,
                                   font=('Georgia', 28),
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
                                   cursor="hand2")
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = tk.Frame(self.root, bg='#1a1a1a', height=100)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame,
                                    text="Status: Initializing...",
                                    font=('Arial', 14),
                                    fg='#96CEB4',
                                    bg='#1a1a1a')
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.prediction_label = tk.Label(self.status_frame,
                                        text="Current word: -",
                                        font=('Arial', 14),
                                        fg='#FFD93D',
                                        bg='#1a1a1a')
        self.prediction_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Bind events
        self.text_display.bind('<Motion>', self.on_mouse_motion)
        self.text_display.bind('<Button-1>', self.on_click)
        self.root.bind('<Left>', lambda e: self.change_text(-1))
        self.root.bind('<Right>', lambda e: self.change_text(1))
        self.root.bind('q', lambda e: self.shutdown())
        
        # Display initial text
        self.display_text()
    
    def display_text(self):
        """Display current text"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', TEST_TEXTS[self.current_text_index])
        
        # Reset predictions
        self.word_predictions.clear()
        
        # Tag all words for tracking
        self._tag_words()
        
        self.text_display.config(state=tk.DISABLED)
    
    def _tag_words(self):
        """Tag each word in the text for tracking"""
        text_content = self.text_display.get('1.0', tk.END)
        
        # Simple word tokenization
        word_start = '1.0'
        word_index = 0
        
        while True:
            # Find next word
            word_start = self.text_display.search(r'\S+', word_start, tk.END, regexp=True)
            if not word_start:
                break
            
            # Find word end
            word_end = self.text_display.search(r'\s', word_start, tk.END)
            if not word_end:
                word_end = tk.END
            
            # Tag this word
            tag_name = f'word_{word_index}'
            self.text_display.tag_add(tag_name, word_start, word_end)
            
            word_index += 1
            word_start = word_end
    
    def on_mouse_motion(self, event):
        """Track mouse movement and update predictions"""
        try:
            # Get word at cursor
            index = self.text_display.index(f"@{event.x},{event.y}")
            
            # Get word boundaries
            word_start = self.text_display.index(f"{index} wordstart")
            word_end = self.text_display.index(f"{index} wordend")
            
            word = self.text_display.get(word_start, word_end).strip()
            
            if word and word != self.current_word:
                self.current_word = word
                self.current_word_index = word_start
                
                # Update prediction for this word
                self._update_word_prediction(word, word_start, word_end)
                
        except Exception as e:
            pass
    
    def _update_word_prediction(self, word, word_start, word_end):
        """Update confusion prediction for current word"""
        with self.lock:
            # Check if we have enough data
            if len(self.eeg_buffer) < 50:
                return
            
            # Get current timestamp
            current_time = time.time()
            
            # Extract features for current word
            features = self._extract_live_features(word, current_time)
            
            if features is not None:
                # Apply model
                features_scaled = self.scaler.transform([features])
                confusion_prob = self.model.predict_proba(features_scaled)[0, 1]
                
                # Store prediction
                self.word_predictions[word_start] = confusion_prob
                
                # Update display
                self._highlight_word(word_start, word_end, confusion_prob)
                
                # Update status
                self.prediction_label.config(
                    text=f"Current: {word[:15]} ({confusion_prob:.2%} confusion)"
                )
    
    def _extract_live_features(self, word, timestamp):
        """Extract features from live data buffers"""
        try:
            # Get recent data
            recent_eeg = list(self.eeg_buffer)[-int(self.window_size * self.sample_rate):]
            recent_fnirs = list(self.fnirs_buffer)[-int(self.window_size * self.sample_rate):] if self.fnirs_buffer else None
            
            if len(recent_eeg) < 10:
                return None
            
            # Convert to arrays
            eeg_array = np.array(recent_eeg)
            fnirs_array = np.array(recent_fnirs) if recent_fnirs else None
            
            # Use feature extractor
            features = self.feature_extractor.extract_features_from_arrays(
                word, eeg_array, fnirs_array
            )
            
            return features
            
        except Exception as e:
            return None
    
    def _highlight_word(self, word_start, word_end, confusion_prob):
        """Highlight word based on confusion probability"""
        # Remove old highlighting
        self.text_display.tag_remove('highlight', word_start, word_end)
        
        # Apply new highlighting based on probability
        if confusion_prob > 0.3:  # Only highlight if above threshold
            color_index = int(confusion_prob * 100)
            color = self.confusion_gradient[min(color_index, 100)]
            
            tag_name = f'confusion_{int(confusion_prob*100)}'
            self.text_display.tag_configure(tag_name, background=color)
            self.text_display.tag_add(tag_name, word_start, word_end)
    
    def on_click(self, event):
        """Handle click feedback"""
        try:
            # Get clicked word
            index = self.text_display.index(f"@{event.x},{event.y}")
            word_start = self.text_display.index(f"{index} wordstart")
            word_end = self.text_display.index(f"{index} wordend")
            word = self.text_display.get(word_start, word_end).strip()
            
            if word:
                # Flash to indicate feedback received
                self.text_display.tag_configure('feedback', background='#ffff00')
                self.text_display.tag_add('feedback', word_start, word_end)
                self.root.after(200, lambda: self.text_display.tag_remove('feedback', word_start, word_end))
                
                print(f"Feedback: Confused by '{word}'")
                
        except Exception as e:
            pass
    
    def change_text(self, direction):
        """Change to next/previous text"""
        self.current_text_index = (self.current_text_index + direction) % len(TEST_TEXTS)
        self.display_text()
    
    def start_muse_receiver(self):
        """Start receiving Muse data"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            self.running = True
            
            receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            receiver_thread.start()
            
            print(f"✅ Muse receiver started on port {self.port}")
            
        except Exception as e:
            print(f"❌ Failed to start Muse receiver: {e}")
    
    def _receiver_loop(self):
        """Receive and process Muse data"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = self._parse_osc_message(data)
                
                if message:
                    self._process_muse_data(message)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def _parse_osc_message(self, data):
        """Parse OSC message (simplified from original)"""
        try:
            # Find address
            addr_end = data.find(b'\x00')
            if addr_end == -1:
                return None
            
            address = data[:addr_end].decode('ascii')
            
            # Parse based on known message types
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                return {'address': address, 'data_type': parts[1], 'raw': data}
            
        except:
            pass
        return None
    
    def _process_muse_data(self, message):
        """Process incoming Muse data"""
        data_type = message['data_type']
        
        with self.lock:
            timestamp = time.time()
            self.timestamps.append(timestamp)
            
            # Simplified processing - in real implementation, parse actual values
            if data_type == 'eeg':
                # Extract 4 EEG values (placeholder)
                self.eeg_buffer.append([0, 0, 0, 0])  # Replace with actual parsing
                
                if len(self.eeg_buffer) == 1:
                    self.status_label.config(text="Status: Receiving EEG data")
                    
            elif data_type == 'optics':
                # Extract 8 fNIRS values (placeholder)
                self.fnirs_buffer.append([0, 0, 0, 0, 0, 0, 0, 0])  # Replace with actual parsing
    
    def start_prediction_loop(self):
        """Start continuous prediction updates"""
        def update_predictions():
            if self.running:
                # Update status
                buffer_size = len(self.eeg_buffer)
                self.status_label.config(
                    text=f"Status: Active | Buffer: {buffer_size} samples"
                )
                
                # Schedule next update
                self.root.after(100, update_predictions)
        
        update_predictions()
    
    def shutdown(self):
        """Shutdown the system"""
        print("\nShutting down...")
        self.running = False
        if self.socket:
            self.socket.close()
        if self.root:
            self.root.destroy()


class FeatureExtractor:
    """Feature extraction for confusion detection"""
    
    def __init__(self, sample_rate=256, eeg_channels=None):
        self.sample_rate = sample_rate
        self.channels = eeg_channels or ['TP9', 'AF7', 'AF8', 'TP10']
        
    def extract_features(self, word, timestamp, timestamps, eeg, fnirs=None, window_size=2.0):
        """Extract features for a word at given timestamp"""
        # Find samples in window
        timestamps = np.array(timestamps)
        start_time = timestamp - window_size/2
        end_time = timestamp + window_size/2
        
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        indices = np.where(mask)[0]
        
        if len(indices) < 10:
            return None
        
        eeg_window = eeg[indices]
        fnirs_window = fnirs[indices] if fnirs is not None else None
        
        return self.extract_features_from_arrays(word, eeg_window, fnirs_window)
    
    def extract_features_from_arrays(self, word, eeg_array, fnirs_array=None):
        """Extract features from data arrays"""
        features = []
        
        # EEG features
        for ch in range(eeg_array.shape[1] if len(eeg_array.shape) > 1 else 1):
            if len(eeg_array.shape) == 1:
                channel_data = eeg_array
            else:
                channel_data = eeg_array[:, ch]
            
            # Time domain features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(np.abs(channel_data)),
                stats.skew(channel_data) if len(channel_data) > 2 else 0,
                stats.kurtosis(channel_data) if len(channel_data) > 3 else 0
            ])
            
            # Frequency domain features
            if len(channel_data) >= 64:
                freqs, psd = signal.welch(channel_data, fs=self.sample_rate, 
                                        nperseg=min(64, len(channel_data)))
                
                # Band powers
                for low, high in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]:
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        features.append(np.log10(np.mean(psd[mask]) + 1e-10))
                    else:
                        features.append(0)
                
                # Peak frequency
                features.append(freqs[np.argmax(psd)])
                
                # Spectral entropy
                psd_norm = psd / (np.sum(psd) + 1e-10)
                entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                features.append(entropy)
            else:
                features.extend([0] * 7)
        
        # fNIRS features if available
        if fnirs_array is not None:
            for ch in range(min(4, fnirs_array.shape[1] if len(fnirs_array.shape) > 1 else 1)):
                if len(fnirs_array.shape) == 1:
                    channel_data = fnirs_array
                else:
                    channel_data = fnirs_array[:, ch]
                
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
    
    def _count_syllables(self, word):
        """Count syllables in word"""
        count = 0
        vowels = "aeiouAEIOU"
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        return max(1, count)


def main():
    """Main function"""
    import sys
    import glob
    
    # Find training data
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for confusion click data
        npz_files = glob.glob("*confusion_clicks*.npz")
        npz_files.extend(glob.glob(os.path.expanduser("~/Downloads/*confusion_clicks*.npz")))
        
        if not npz_files:
            print("❌ No training data found!")
            print("Please specify a confusion_clicks NPZ file")
            return
        
        if len(npz_files) == 1:
            filepath = npz_files[0]
        else:
            print("Available training data:")
            for i, f in enumerate(npz_files):
                print(f"  {i+1}. {os.path.basename(f)}")
            choice = int(input("Select file: ")) - 1
            filepath = npz_files[choice]
    
    print(f"\n{'='*60}")
    print("LIVE CONFUSION DETECTION SYSTEM")
    print(f"{'='*60}")
    
    # Create detector
    detector = LiveConfusionDetector()
    
    # Load training data
    detector.load_training_data(filepath)
    
    # Train model
    detector.train_model()
    
    # Start live detection
    print("\n🚀 Starting live detection...")
    print("Make sure Muse is streaming data to port 8052")
    
    try:
        detector.start_live_detection()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()