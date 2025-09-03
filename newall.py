import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import cv2
import time
import json
import pickle
import socket
import struct
import sys
from datetime import datetime
from collections import deque
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

class ConfusionTrainer:
    def __init__(self, port=5000):
        self.root = tk.Tk()
        self.root.title("Confusion Recognition Trainer - Muse S")
        self.root.geometry("1200x800")
        
        # OSC/UDP setup
        self.port = port
        self.socket = None
        self.running = True
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Data buffers - Muse S Athena format
        self.buffer_size = 500  # ~2 seconds at 256Hz
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # EEG channels for Athena
        self.eeg_channels = {
            'TP9': deque(maxlen=self.buffer_size),
            'AF7': deque(maxlen=self.buffer_size),
            'AF8': deque(maxlen=self.buffer_size),
            'TP10': deque(maxlen=self.buffer_size)
        }
        
        # fNIRS/Optics channels
        self.fnirs_channels = {
            'Ch1_norm': deque(maxlen=self.buffer_size),
            'Ch2_norm': deque(maxlen=self.buffer_size),
            'Ch3_norm': deque(maxlen=self.buffer_size),
            'Ch4_norm': deque(maxlen=self.buffer_size),
            'Ch1_raw': deque(maxlen=self.buffer_size),
            'Ch2_raw': deque(maxlen=self.buffer_size),
            'Ch3_raw': deque(maxlen=self.buffer_size),
            'Ch4_raw': deque(maxlen=self.buffer_size)
        }
        
        # Motion channels
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
        
        # Last known values for recording
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Training data
        self.head_position = [0, 0]
        self.current_word_index = 0
        self.confusion_moments = []
        self.training_data = []
        self.is_recording = False
        self.lock = threading.Lock()
        
        # Head tracking
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.baseline_head_pos = None
        
        # Model components
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10)
        self.scaler = StandardScaler()
        
        # Sample texts
        self.texts = [
            "The quantum entanglement phenomenon demonstrates non-local correlations between particles.",
            "Neuroplasticity enables synaptic pruning through hebbian mechanisms in cortical regions.",
            "The stochastic gradient descent algorithm optimizes convex functions iteratively.",
            "Photosynthesis converts electromagnetic radiation into chemical energy via chlorophyll.",
            "Epistemological frameworks underpin phenomenological investigations of consciousness."
        ]
        self.current_text_idx = 0
        self.words = []
        
        self.setup_ui()
        self.start_osc_receiver()
        self.start_head_tracking()
        
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
            
            # Parse address
            address, offset = self.parse_osc_string(data, offset)
            if not address:
                return None
            
            # Add leading / if missing
            if not address.startswith('/'):
                address = '/' + address
            
            # Parse type tags
            type_tags, offset = self.parse_osc_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            type_tags = type_tags[1:]
            
            # Parse arguments
            args = []
            for tag in type_tags:
                if tag == 'f':  # float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 'i':  # int
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
            # Parse address: /username/datatype
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                # EEG data - 4 floats for Athena
                if data_type == 'eeg' and len(args) == 4:
                    self.timestamps.append(timestamp)
                    
                    channels = ['TP9', 'AF7', 'AF8', 'TP10']
                    for ch, val in zip(channels, args):
                        self.eeg_channels[ch].append(val)
                    self.eeg_packet_count += 1
                    self.last_eeg_data = args
                    
                    # Debug first few packets
                    if self.eeg_packet_count <= 3:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                # fNIRS/Optics data - 8 floats
                elif data_type == 'optics' and len(args) == 8:
                    norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
                    raw_channels = ['Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw']
                    
                    for i, (ch, val) in enumerate(zip(norm_channels + raw_channels, args)):
                        self.fnirs_channels[ch].append(val)
                    
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                    
                    if self.fnirs_packet_count <= 3:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}")
                
                # Accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    channels = ['acc_x', 'acc_y', 'acc_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[:3] = args
                
                # Gyroscope data
                elif data_type == 'gyro' and len(args) == 3:
                    channels = ['gyro_x', 'gyro_y', 'gyro_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[3:] = args
                
                # DRL/REF data
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
                
                # Try to parse as OSC message
                message = self.parse_osc_message(data)
                if message:
                    self.process_osc_message(message)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def start_osc_receiver(self):
        """Start OSC/UDP receiver"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            
            print(f"OSC receiver listening on port {self.port}")
            
            # Start receiver thread
            receiver_thread = threading.Thread(target=self.receiver_loop, daemon=True)
            receiver_thread.start()
            
        except Exception as e:
            print(f"Failed to start OSC receiver: {e}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text display area
        text_frame = ttk.LabelFrame(main_frame, text="Reading Material", padding="10")
        text_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.text_display = tk.Text(text_frame, height=10, width=80, font=('Arial', 14), wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Muse S Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        self.status_label = ttk.Label(status_frame, text="Ready to start")
        self.status_label.pack()
        
        self.word_label = ttk.Label(status_frame, text="Current word: -", font=('Arial', 12, 'bold'))
        self.word_label.pack(pady=5)
        
        self.eeg_label = ttk.Label(status_frame, text="EEG: Waiting for data...")
        self.eeg_label.pack()
        
        self.fnirs_label = ttk.Label(status_frame, text="fNIRS: Waiting for data...")
        self.fnirs_label.pack()
        
        self.packet_label = ttk.Label(status_frame, text="Packets: 0")
        self.packet_label.pack()
        
        self.head_label = ttk.Label(status_frame, text="Head Position: -")
        self.head_label.pack()
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="Load Text", command=self.load_text).pack(pady=2, fill=tk.X)
        ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording).pack(pady=2, fill=tk.X)
        ttk.Button(control_frame, text="Train Model", command=self.train_model).pack(pady=2, fill=tk.X)
        ttk.Button(control_frame, text="Save Session", command=self.save_session).pack(pady=2, fill=tk.X)
        ttk.Button(control_frame, text="Load Session", command=self.load_session).pack(pady=2, fill=tk.X)
        
        ttk.Label(control_frame, text=f"\nOSC Port: {self.port}", font=('Arial', 10)).pack()
        ttk.Label(control_frame, text="Press 's' = Mark Confusion", font=('Arial', 10, 'bold')).pack()
        ttk.Label(control_frame, text="Press Space = Calibrate Head", font=('Arial', 10)).pack()
        
        # Data display
        data_frame = ttk.LabelFrame(main_frame, text="Collected Data", padding="10")
        data_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=8, width=80)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind keys
        self.root.bind('<s>', self.mark_confusion)
        self.root.bind('<space>', self.calibrate_head)
        
        # Update display periodically
        self.update_display()
    
    def update_display(self):
        """Update the display with current sensor values"""
        with self.lock:
            # EEG status
            if len(self.eeg_channels['TP9']) > 0:
                latest_eeg = {ch: vals[-1] if vals else 0 for ch, vals in self.eeg_channels.items()}
                eeg_text = f"EEG: TP9={latest_eeg['TP9']:.1f} AF7={latest_eeg['AF7']:.1f}"
                self.eeg_label.config(text=eeg_text)
            
            # fNIRS status
            if len(self.fnirs_channels['Ch1_norm']) > 0:
                latest_fnirs = {ch: vals[-1] if vals else 0 for ch, vals in self.fnirs_channels.items()}
                fnirs_text = f"fNIRS: Ch1={latest_fnirs['Ch1_norm']:.2f} Ch2={latest_fnirs['Ch2_norm']:.2f}"
                self.fnirs_label.config(text=fnirs_text)
            
            # Packet count
            self.packet_label.config(text=f"Packets: {self.packet_count} (EEG:{self.eeg_packet_count} fNIRS:{self.fnirs_packet_count})")
        
        # Schedule next update
        self.root.after(100, self.update_display)
    
    def load_text(self):
        """Load next text and parse words"""
        self.text_display.delete('1.0', tk.END)
        current_text = self.texts[self.current_text_idx % len(self.texts)]
        self.text_display.insert('1.0', current_text)
        
        # Parse words and their positions
        self.words = current_text.split()
        self.current_word_index = 0
        self.highlight_current_word()
        
        self.current_text_idx += 1
        self.status_label.config(text=f"Text {self.current_text_idx} loaded")
        
    def highlight_current_word(self):
        """Highlight the word being looked at"""
        if not self.words:
            return
            
        # Clear previous highlights
        self.text_display.tag_remove('current', '1.0', tk.END)
        self.text_display.tag_configure('current', background='yellow')
        
        # Find and highlight current word
        word_to_find = self.words[min(self.current_word_index, len(self.words)-1)]
        start_idx = '1.0'
        
        # Simple word position tracking
        for i in range(self.current_word_index + 1):
            pos = self.text_display.search(self.words[min(i, len(self.words)-1)], start_idx, tk.END)
            if pos and i == self.current_word_index:
                end_pos = f"{pos}+{len(word_to_find)}c"
                self.text_display.tag_add('current', pos, end_pos)
                self.word_label.config(text=f"Current word: {word_to_find}")
                break
            if pos:
                start_idx = f"{pos}+{len(self.words[i])}c"
    
    def start_head_tracking(self):
        """Start head tracking thread"""
        head_thread = threading.Thread(target=self.track_head, daemon=True)
        head_thread.start()
    
    def track_head(self):
        """Track head position for reading position estimation"""
        self.cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                center_x = x + w // 2
                center_y = y + h // 2
                
                if self.baseline_head_pos is None:
                    self.baseline_head_pos = [center_x, center_y]
                
                # Calculate relative position
                rel_x = center_x - self.baseline_head_pos[0]
                rel_y = center_y - self.baseline_head_pos[1]
                self.head_position = [rel_x, rel_y]
                
                # Estimate reading position from head movement
                self.estimate_reading_position(rel_x, rel_y)
                
                self.head_label.config(text=f"Head: ({rel_x:.0f}, {rel_y:.0f})")
                
            time.sleep(0.03)  # ~30 FPS
    
    def estimate_reading_position(self, rel_x, rel_y):
        """Estimate which word is being looked at based on head position"""
        if not self.words:
            return
            
        # Simple mapping: rightward movement = forward in text
        pixels_per_word = 20
        
        # Calculate word index from head position
        word_offset = int(rel_x / pixels_per_word)
        new_index = max(0, min(len(self.words) - 1, word_offset))
        
        if new_index != self.current_word_index:
            self.current_word_index = new_index
            self.highlight_current_word()
    
    def calibrate_head(self, event):
        """Calibrate head position baseline (spacebar)"""
        self.baseline_head_pos = None
        self.status_label.config(text="Head position calibrated")
    
    def toggle_recording(self):
        """Start/stop data recording"""
        self.is_recording = not self.is_recording
        status = "Recording..." if self.is_recording else "Stopped"
        self.status_label.config(text=status)
    
    def mark_confusion(self, event):
        """Mark current moment as confusion (s key)"""
        if not self.is_recording or not self.words:
            return
            
        # Collect features from real Muse S data
        features = self.extract_features()
        
        if features is not None:
            # Store confusion moment
            confusion_data = {
                'timestamp': time.time(),
                'word': self.words[self.current_word_index],
                'word_index': self.current_word_index,
                'features': features,
                'label': 1  # 1 for confused
            }
            
            self.confusion_moments.append(confusion_data)
            self.training_data.append((features, 1))
            
            # Also collect some non-confusion samples around it
            for _ in range(3):
                time.sleep(0.1)
                feat = self.extract_features()
                if feat is not None:
                    self.training_data.append((feat, 0))
            
            # Update display
            self.data_text.insert(tk.END, f"Confusion marked at: {self.words[self.current_word_index]}\n")
            self.data_text.see(tk.END)
        else:
            self.status_label.config(text="No EEG data available yet")
    
    def extract_features(self):
        """Extract features from current Muse S sensor data"""
        with self.lock:
            features = []
            
            # Check if we have data
            if len(self.eeg_channels['TP9']) < 10:
                return None
            
            # EEG features (4 channels)
            for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
                if len(self.eeg_channels[ch]) > 0:
                    recent = list(self.eeg_channels[ch])[-50:]  # Last ~200ms at 256Hz
                    features.append(np.mean(recent))
                    features.append(np.std(recent))
                    features.append(np.max(recent) - np.min(recent))  # Range
                else:
                    features.extend([0, 0, 0])
            
            # fNIRS features (normalized channels)
            for ch in ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']:
                if len(self.fnirs_channels[ch]) > 0:
                    recent = list(self.fnirs_channels[ch])[-10:]
                    features.append(np.mean(recent))
                    features.append(np.std(recent))
                else:
                    features.extend([0, 0])
            
            # Motion features (acceleration magnitude)
            if len(self.motion_channels['acc_x']) > 0:
                acc_x = list(self.motion_channels['acc_x'])[-10:]
                acc_y = list(self.motion_channels['acc_y'])[-10:]
                acc_z = list(self.motion_channels['acc_z'])[-10:]
                acc_mag = np.sqrt(np.array(acc_x)**2 + np.array(acc_y)**2 + np.array(acc_z)**2)
                features.append(np.mean(acc_mag))
                features.append(np.std(acc_mag))
            else:
                features.extend([0, 0])
            
            # Head position features
            features.extend(self.head_position)
            features.append(abs(self.head_position[0]))  # Movement magnitude
            
            # Reading context
            features.append(self.current_word_index)
            
            return np.array(features)
    
    def train_model(self):
        """Train the confusion recognition model"""
        if len(self.training_data) < 10:
            self.status_label.config(text="Need more data (10+ samples)")
            return
            
        X = np.array([f for f, _ in self.training_data])
        y = np.array([l for _, l in self.training_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Simple validation
        accuracy = self.model.score(X_scaled, y)
        
        self.status_label.config(text=f"Model trained! Accuracy: {accuracy:.2%}")
        self.data_text.insert(tk.END, f"\nModel trained with {len(X)} samples\n")
        
        # Feature importance
        importances = self.model.feature_importances_
        feature_names = ['EEG_TP9_mean', 'EEG_TP9_std', 'EEG_TP9_range',
                        'EEG_AF7_mean', 'EEG_AF7_std', 'EEG_AF7_range',
                        'EEG_AF8_mean', 'EEG_AF8_std', 'EEG_AF8_range',
                        'EEG_TP10_mean', 'EEG_TP10_std', 'EEG_TP10_range',
                        'fNIRS_Ch1_mean', 'fNIRS_Ch1_std',
                        'fNIRS_Ch2_mean', 'fNIRS_Ch2_std',
                        'fNIRS_Ch3_mean', 'fNIRS_Ch3_std',
                        'fNIRS_Ch4_mean', 'fNIRS_Ch4_std',
                        'Motion_mag_mean', 'Motion_mag_std',
                        'Head_X', 'Head_Y', 'Head_mag', 'Word_index']
        
        # Show top 5 important features
        indices = np.argsort(importances)[-5:]
        self.data_text.insert(tk.END, "\nTop 5 important features:\n")
        for i in indices:
            self.data_text.insert(tk.END, f"  {feature_names[i]}: {importances[i]:.3f}\n")
    
    def save_session(self):
        """Save training data and model"""
        session_data = {
            'training_data': [(f.tolist(), l) for f, l in self.training_data],
            'confusion_moments': self.confusion_moments,
            'timestamp': datetime.now().isoformat(),
            'eeg_packets': self.eeg_packet_count,
            'fnirs_packets': self.fnirs_packet_count
        }
        
        filename = f"muse_confusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
            
        # Save model
        if len(self.training_data) > 0:
            model_file = f"muse_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump((self.model, self.scaler), f)
                
        self.status_label.config(text=f"Session saved to {filename}")
    
    def load_session(self):
        """Load previous training data"""
        files = [f for f in os.listdir('.') if f.startswith('muse_confusion')]
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)
                self.training_data = [(np.array(f), l) for f, l in data['training_data']]
                self.confusion_moments = data['confusion_moments']
                
            self.status_label.config(text=f"Loaded {len(self.training_data)} samples")
            self.data_text.insert(tk.END, f"\nLoaded session from {files[-1]}\n")
    
    def cleanup(self):
        """Cleanup on exit"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.cap:
            self.cap.release()
    
    def run(self):
        """Start the application"""
        self.load_text()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    # Port 5000 is default for Mind Monitor and similar apps
    app = ConfusionTrainer(port=5000)
    app.run()