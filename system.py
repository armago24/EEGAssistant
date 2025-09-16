#!/usr/bin/env python3
"""
Live Word-Level Confusion Detection System - Rebuilt Version
Simplified and debugged for reliable operation
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
from tkinter import filedialog
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Test texts
TEST_TEXTS = [
"""Quantum entanglement represents a correlation between quantum systems that persists regardless of spatial separation. When two particles become entangled through interaction, measurements performed on one particle instantaneously affect the quantum state of its partner, a phenomenon Einstein famously derided as "spooky action at a distance." This non-local correlation cannot transmit classical information faster than light, preserving relativistic causality while challenging classical intuitions about locality and realism.""",

"""The renormalization group approach reveals how physical systems behave across different length scales. Starting from microscopic degrees of freedom, successive coarse-graining transformations generate an effective description at larger scales. Fixed points of these transformations correspond to scale-invariant theories, while relevant, marginal, and irrelevant operators determine the system's macroscopic behavior."""
]

class SimpleLiveDetector:
    def __init__(self):
        """Initialize the simplified live detection system"""
        print("\n=== INITIALIZING LIVE DETECTOR ===")
        
        # Model components
        self.model = None
        self.scaler = None
        self.sample_rate = 256
        
        # Network settings
        self.port = 8052
        self.socket = None
        self.running = False
        
        # Data buffers
        self.eeg_buffer = deque(maxlen=1000)
        self.fnirs_buffer = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # UI components
        self.root = None
        self.text_widget = None
        self.status_label = None
        self.current_text_index = 0
        
        # Tracking
        self.current_word = ""
        self.last_word = ""
        self.word_positions = {}  # word -> [(start, end), ...]
        
        # Debug flags
        self.debug_mode = True
        self.test_mode = False  # For testing without real EEG data
        
    def load_and_train(self, filepath):
        """Load training data and train model"""
        print("\n=== LOADING TRAINING DATA ===")
        
        # Load data
        data = np.load(filepath, allow_pickle=True)
        
        # Extract basic info
        print(f"File: {filepath}")
        print(f"Duration: {(data['timestamps'][-1] - data['timestamps'][0]):.1f}s")
        print(f"Samples: {len(data['timestamps'])}")
        print(f"Events: {len(data['event_timestamps'])}")
        
        # Build simple feature extractor
        self.feature_extractor = SimpleFeatureExtractor()
        
        # Create training dataset
        X, y = self._create_simple_dataset(data)
        
        if len(X) == 0:
            raise ValueError("No training data extracted!")
        
        # Train model
        print(f"\n=== TRAINING MODEL ===")
        print(f"Training samples: {len(X)}")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Simple binary classification
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        
        # Check accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = np.mean(y_pred == y)
        print(f"Training accuracy: {accuracy:.3f}")
        
        return True
    
    def _create_simple_dataset(self, data):
        """Create simplified training dataset"""
        X = []
        y = []
        
        # Process confusion events
        confused_times = set()
        if 'event_timestamps' in data:
            for t in data['event_timestamps']:
                confused_times.add(t)
        
        print(f"Found {len(confused_times)} confusion events")
        
        # Sample data points
        timestamps = data['timestamps']
        eeg = data['eeg']
        
        # Take samples around confusion events
        for conf_time in confused_times:
            # Find closest timestamp
            idx = np.argmin(np.abs(timestamps - conf_time))
            
            # Extract window
            start = max(0, idx - 128)
            end = min(len(timestamps), idx + 128)
            
            if end - start > 100:
                features = self._extract_simple_features(eeg[start:end])
                if features is not None:
                    X.append(features)
                    y.append(1)  # Confused
        
        # Add baseline samples
        num_baseline = len(X) * 2
        step = len(timestamps) // num_baseline
        
        for i in range(0, len(timestamps) - 256, step):
            # Check not near confusion
            window_time = timestamps[i + 128]
            if not any(abs(window_time - ct) < 3.0 for ct in confused_times):
                features = self._extract_simple_features(eeg[i:i+256])
                if features is not None:
                    X.append(features)
                    y.append(0)  # Not confused
        
        return np.array(X), np.array(y)
    
    def _extract_simple_features(self, eeg_window):
        """Extract simple features from EEG window"""
        if len(eeg_window) < 100:
            return None
        
        features = []
        eeg_array = np.array(eeg_window)
        
        # Basic stats per channel
        for ch in range(4):  # 4 EEG channels
            if eeg_array.shape[1] > ch:
                channel_data = eeg_array[:, ch]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.max(np.abs(channel_data))
                ])
        
        return np.array(features)
    
    def create_ui(self):
        """Create simple UI"""
        print("\n=== CREATING UI ===")
        
        self.root = tk.Tk()
        self.root.title("Live Confusion Detection")
        self.root.geometry("1000x700")
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing...", font=('Arial', 14))
        self.status_label.pack(pady=5)
        
        # Debug label
        self.debug_label = tk.Label(main_frame, text="Debug: Starting...", font=('Arial', 10))
        self.debug_label.pack(pady=5)
        
        # Text widget
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_widget = tk.Text(
            text_frame,
            font=('Georgia', 24),
            wrap=tk.WORD,
            height=10,
            width=50,
            bg='white',
            fg='black'
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Test Highlight", command=self.test_highlight).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Previous Text", command=lambda: self.change_text(-1)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Next Text", command=lambda: self.change_text(1)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Toggle Test Mode", command=self.toggle_test_mode).pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(main_frame, text="Move mouse over text to see predictions", font=('Arial', 12))
        instructions.pack(pady=5)
        
        # Load initial text
        self.load_text()
        
        # Set up cursor tracking
        self.setup_cursor_tracking()
        
        print("UI created successfully")
    
    def setup_cursor_tracking(self):
        """Set up cursor position tracking"""
        print("\n=== SETTING UP CURSOR TRACKING ===")
        
        # Track cursor position with polling
        def poll_cursor():
            try:
                # Get cursor position
                x = self.root.winfo_pointerx() - self.text_widget.winfo_rootx()
                y = self.root.winfo_pointery() - self.text_widget.winfo_rooty()
                
                # Check if inside text widget
                if 0 <= x <= self.text_widget.winfo_width() and 0 <= y <= self.text_widget.winfo_height():
                    # Get word at position
                    pos = self.text_widget.index(f"@{x},{y}")
                    word_range = self.text_widget.tag_nextrange("word", pos)
                    
                    if word_range:
                        start, end = word_range
                        word = self.text_widget.get(start, end)
                        
                        if word != self.current_word:
                            self.current_word = word
                            self.on_word_hover(word, start, end)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Cursor tracking error: {e}")
            
            # Schedule next poll
            if self.running:
                self.root.after(20, poll_cursor)  # 50 Hz polling
        
        # Start polling
        self.running = True
        poll_cursor()
        print("Cursor tracking started")
    
    def toggle_heatmap_mode(self):
        """Toggle between heatmap and single word highlighting"""
        self.heatmap_mode = not self.heatmap_mode
        mode = "Heatmap" if self.heatmap_mode else "Single Word"
        print(f"\nHighlight mode: {mode}")
        self.debug_label.config(text=f"Mode: {mode}")
        
        # Clear existing highlights
        self.clear_highlights()
    
    def clear_highlights(self):
        """Clear all word highlights"""
        for tag in self.text_widget.tag_names():
            if tag.startswith("word_") or tag == "current_word":
                self.text_widget.tag_remove(tag, '1.0', tk.END)
        self.word_highlights.clear()
        print("Cleared all highlights")
    
    def load_text(self):
        """Load current text and tag words"""
        print(f"\nLoading text {self.current_text_index + 1}...")
        
        # Clear previous highlights
        self.clear_highlights()
        
        # Clear text
        self.text_widget.delete('1.0', tk.END)
        
        # Insert new text
        text = TEST_TEXTS[self.current_text_index]
        self.text_widget.insert('1.0', text)
        
        # Tag each word for tracking
        self.word_positions.clear()
        current_pos = '1.0'
        word_count = 0
        
        while True:
            # Find next word
            word_start = self.text_widget.search(r'\S+', current_pos, tk.END, regexp=True)
            if not word_start:
                break
            
            # Find word end
            word_end = self.text_widget.search(r'\s', word_start, tk.END)
            if not word_end:
                word_end = tk.END
            
            # Get word
            word = self.text_widget.get(word_start, word_end)
            
            # Tag it
            tag_name = f"word"
            self.text_widget.tag_add(tag_name, word_start, word_end)
            
            # Store position
            if word not in self.word_positions:
                self.word_positions[word] = []
            self.word_positions[word].append((word_start, word_end))
            
            word_count += 1
            current_pos = word_end
        
        print(f"Tagged {word_count} words")
    
    def on_word_hover(self, word, start, end):
        """Handle hovering over a word"""
        if word == self.last_word:
            return
        
        self.last_word = word
        
        # Update debug info
        self.debug_label.config(text=f"Hovering: '{word}' at {start}")
        
        # Get prediction
        if self.test_mode or len(self.eeg_buffer) > 50:
            prob = self.predict_confusion(word)
            self.highlight_word(start, end, prob)
            
            # Update status
            self.status_label.config(text=f"Word: '{word}' - Confusion: {prob:.1%}")
        else:
            self.status_label.config(text=f"Word: '{word}' - Waiting for EEG data...")
    
    def predict_confusion(self, word):
        """Predict confusion probability for word"""
        if self.test_mode:
            # Fake predictions for testing
            if 'quantum' in word.lower():
                return 0.8
            elif 'entanglement' in word.lower():
                return 0.7
            elif 'Einstein' in word.lower():
                return 0.6
            else:
                return np.random.random() * 0.5
        
        # Real prediction
        try:
            # Get recent EEG data
            with self.lock:
                if len(self.eeg_buffer) < 100:
                    return 0.0
                
                recent_eeg = list(self.eeg_buffer)[-256:]
            
            # Extract features
            features = self._extract_simple_features(recent_eeg)
            if features is None:
                return 0.0
            
            # Apply model
            features_scaled = self.scaler.transform([features])
            prob = self.model.predict_proba(features_scaled)[0, 1]
            
            return prob
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    
    def highlight_word(self, start, end, confusion_prob):
        """Apply highlighting to word"""
        # Remove old highlights
        self.text_widget.tag_remove("highlight", start, end)
        
        # Determine color
        if confusion_prob > 0.7:
            color = '#ff4444'  # Red
        elif confusion_prob > 0.5:
            color = '#ff8844'  # Orange
        elif confusion_prob > 0.3:
            color = '#ffdd44'  # Yellow
        else:
            color = '#88ff88'  # Light green
        
        # Apply highlight
        self.text_widget.tag_add("highlight", start, end)
        self.text_widget.tag_config("highlight", background=color)
        
        if self.debug_mode and confusion_prob > 0.5:
            print(f"Highlighted '{self.current_word}' with {confusion_prob:.1%} confusion")
    
    def test_highlight(self):
        """Test highlighting functionality"""
        print("\n=== TESTING HIGHLIGHTS ===")
        
        # Get first 5 words
        colors = ['#ff0000', '#ff8800', '#ffff00', '#88ff00', '#00ff00']
        
        pos = '1.0'
        for i, color in enumerate(colors):
            # Find next word
            word_start = self.text_widget.search(r'\S+', pos, tk.END, regexp=True)
            if not word_start:
                break
            
            word_end = self.text_widget.search(r'\s', word_start, tk.END)
            if not word_end:
                word_end = f"{word_start} lineend"
            
            word = self.text_widget.get(word_start, word_end)
            
            # Apply color
            tag = f'test_{i}'
            self.text_widget.tag_add(tag, word_start, word_end)
            self.text_widget.tag_config(tag, background=color)
            
            print(f"Colored '{word}' with {color}")
            
            pos = word_end
        
        print("Test complete - you should see 5 colored words")
    
    def toggle_test_mode(self):
        """Toggle test mode (no EEG required)"""
        self.test_mode = not self.test_mode
        status = "ON" if self.test_mode else "OFF"
        print(f"\nTest mode: {status}")
        self.debug_label.config(text=f"Test mode: {status}")
    
    def change_text(self, direction):
        """Change to next/previous text"""
        self.current_text_index = (self.current_text_index + direction) % len(TEST_TEXTS)
        self.load_text()
    
    def start_receiver(self):
        """Start receiving EEG data"""
        print(f"\n=== STARTING RECEIVER (Port {self.port}) ===")
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            
            # Start receiver thread
            receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
            receiver_thread.start()
            
            print("Receiver started successfully")
            
        except Exception as e:
            print(f"Failed to start receiver: {e}")
            print("Continue in test mode")
            self.test_mode = True
    
    def _receive_loop(self):
        """Receive EEG data"""
        packet_count = 0
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                
                # Parse OSC message
                message = self._parse_osc(data)
                if message and 'eeg' in message['address']:
                    with self.lock:
                        self.eeg_buffer.append(message['args'])
                        self.timestamps.append(time.time())
                    
                    packet_count += 1
                    if packet_count % 100 == 0:
                        print(f"Received {packet_count} EEG packets")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receiver error: {e}")
    
    def _parse_osc(self, data):
        """Parse OSC message"""
        try:
            # Find address
            addr_end = data.find(b'\x00')
            if addr_end == -1:
                return None
            
            address = data[:addr_end].decode('ascii')
            
            # Simple parse for EEG floats
            if 'eeg' in address:
                # Skip to float data (simplified)
                offset = ((addr_end + 4) // 4) * 4 + 8  # Skip padding and type tags
                
                args = []
                for i in range(4):  # 4 EEG channels
                    if offset + 4 <= len(data):
                        value = struct.unpack('>f', data[offset:offset+4])[0]
                        args.append(value)
                        offset += 4
                
                if len(args) == 4:
                    return {'address': address, 'args': args}
        
        except:
            pass
        
        return None
    
    def run(self):
        """Run the system"""
        print("\n=== STARTING LIVE DETECTION ===")
        
        # Start receiver
        self.start_receiver()
        
        # Run UI
        print("\nSystem ready!")
        print("- Move mouse over text to see predictions")
        print("- Click 'Test Highlight' to verify colors work")
        print("- Click 'Toggle Test Mode' to use fake predictions")
        
        self.root.mainloop()


class SimpleFeatureExtractor:
    """Simplified feature extraction"""
    
    def extract(self, eeg_window):
        """Extract basic features"""
        features = []
        
        for ch in range(4):
            if ch < eeg_window.shape[1]:
                channel = eeg_window[:, ch]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.max(np.abs(channel))
                ])
        
        return np.array(features)


def main():
    """Main function"""
    import sys
    import glob
    
    print("=" * 60)
    print("LIVE CONFUSION DETECTION - SIMPLIFIED VERSION")
    print("=" * 60)
    
    # Find training data
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        npz_files = glob.glob("*confusion*.npz")
        npz_files.extend(glob.glob(os.path.expanduser("~/Downloads/*confusion*.npz")))
        
        if not npz_files:
            print("No training data found!")
            return
        
        filepath = npz_files[0]
        print(f"Using: {filepath}")
    
    # Create detector
    detector = SimpleLiveDetector()
    
    # Load and train
    try:
        detector.load_and_train(filepath)
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    # Create UI
    detector.create_ui()
    
    # Run
    detector.run()


if __name__ == "__main__":
    main()