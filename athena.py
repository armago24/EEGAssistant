import socket
import struct
from collections import deque, defaultdict
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.fft import fft, fftfreq

class EEGfNIRSVisualizerStandalone:
    def __init__(self, port=8052, buffer_size=1000, update_interval=50, window_duration=10):
        # Parser attributes
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.running = False
        self.data_format_detected = False
        
        # Data buffers
        self.eeg_channels = {}
        self.fnirs_channels = {}
        self.timestamps = deque(maxlen=buffer_size)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Debug mode
        self.debug = False
        
        # Visualizer attributes
        self.update_interval = update_interval  # milliseconds
        self.window_duration = window_duration  # seconds to display
        
        # Setup matplotlib with dark theme
        plt.style.use('dark_background')
        
        # Data line objects
        self.eeg_lines = {}
        self.fnirs_lines = {}
        self.motion_lines = {}
        self.fft_lines = {}
        
        # Color schemes
        self.eeg_colors = {
            'TP9': '#FF6B6B',    # Red
            'AF7': '#4ECDC4',    # Teal
            'AF8': '#45B7D1',    # Blue
            'TP10': '#96CEB4',   # Green
            'DRL': '#FFD93D',    # Yellow
            'REF': '#FF8B94'     # Pink
        }
        
        self.fnirs_colors = plt.cm.plasma(np.linspace(0, 1, 8))
        self.motion_colors = {
            'acc_x': '#E74C3C', 'acc_y': '#2ECC71', 'acc_z': '#3498DB',
            'gyro_x': '#F39C12', 'gyro_y': '#9B59B6', 'gyro_z': '#1ABC9C'
        }
        
        # Data buffers for statistics
        self.sample_times = deque(maxlen=100)
        self.last_update_time = time.time()
        
        # Y-axis scaling parameters
        self.initial_data_received = False
        
        # FFT parameters
        self.sampling_rate = 256  # Muse S default sampling rate
        self.fft_window_samples = 512  # Samples for FFT calculation
    
    def parse_osc_string(self, data, offset):
        """Parse a null-terminated, 4-byte aligned string from OSC data"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        # Advance to next 4-byte boundary
        offset = ((end + 4) // 4) * 4
        return string, offset
    
    def parse_osc_message(self, data):
        """Manually parse an OSC message"""
        try:
            offset = 0
            
            # Parse address pattern (e.g., "/Aryan/eeg")
            address, offset = self.parse_osc_string(data, offset)
            if not address:
                return None
            
            # Parse type tag string (e.g., ",ffff")
            type_tags, offset = self.parse_osc_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            # Remove the leading comma
            type_tags = type_tags[1:]
            
            # Parse arguments based on type tags
            args = []
            for tag in type_tags:
                if tag == 'f':  # 32-bit float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 'i':  # 32-bit int
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>i', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 's':  # string
                    string, offset = self.parse_osc_string(data, offset)
                    args.append(string)
                # Add more types as needed
            
            return {
                'address': address,
                'args': args
            }
            
        except Exception as e:
            if self.debug:
                print(f"OSC parse error: {e}")
            return None
    
    def start_receiving(self):
        """Start UDP receiver thread"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)  # Non-blocking with timeout
            
            self.running = True
            self.receiver_thread = threading.Thread(target=self._receive_loop)
            self.receiver_thread.start()
            
            print(f"Started listening on UDP port {self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to start receiver: {e}")
            return False
    
    def _receive_loop(self):
        """Main receiver loop"""
        packet_count = 0
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                packet_count += 1
                
                if self.debug and packet_count % 100 == 0:
                    print(f"Packet {packet_count}: {len(data)} bytes from {addr}")
                    # Print hex dump of first few bytes
                    hex_str = ' '.join(f'{b:02x}' for b in data[:32])
                    print(f"  Data: {hex_str}...")
                
                # Try to parse as OSC
                message = self.parse_osc_message(data)
                if message:
                    self._process_osc_message(message)
                    
                    # Mark that we've successfully detected the format
                    if not self.data_format_detected:
                        self.data_format_detected = True
                        print(f"OSC data format detected! Address: {message['address']}")
                        self._print_detected_channels()
                else:
                    # Try to parse as OSC bundle (starts with "#bundle\0")
                    if data.startswith(b'#bundle\x00'):
                        self._parse_osc_bundle(data)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.debug:
                    print(f"Receiver error: {e}")
    
    def _parse_osc_bundle(self, data):
        """Parse an OSC bundle containing multiple messages"""
        try:
            if not data.startswith(b'#bundle\x00'):
                return
            
            # Skip "#bundle\0" (8 bytes) and timestamp (8 bytes)
            offset = 16
            
            while offset < len(data):
                # Read message size (4 bytes)
                if offset + 4 > len(data):
                    break
                    
                msg_size = struct.unpack('>i', data[offset:offset+4])[0]
                offset += 4
                
                if offset + msg_size > len(data):
                    break
                
                # Parse the message
                msg_data = data[offset:offset+msg_size]
                message = self.parse_osc_message(msg_data)
                if message:
                    self._process_osc_message(message)
                
                offset += msg_size
                
        except Exception as e:
            if self.debug:
                print(f"Bundle parse error: {e}")
                
    def _process_osc_message(self, message):
        """Process a parsed OSC message"""
        address = message['address']
        args = message['args']
        timestamp = time.time()
        
        with self.lock:
            # Store timestamp
            self.timestamps.append(timestamp)
            
            # Parse based on address pattern
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                username = parts[0]  # e.g., "Aryan"
                data_type = parts[1]  # e.g., "eeg", "acc", etc.
                
                if self.debug and len(self.timestamps) % 100 == 1:
                    print(f"  OSC: {address} -> {args}")
                
                # Process EEG data (4 channels: TP9, AF7, AF8, TP10)
                if data_type == 'eeg' and len(args) == 4:
                    channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
                    for name, value in zip(channel_names, args):
                        if name not in self.eeg_channels:
                            self.eeg_channels[name] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[name].append(value)
                
                # Process accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    axes = ['acc_x', 'acc_y', 'acc_z']
                    for axis, value in zip(axes, args):
                        if axis not in self.eeg_channels:
                            self.eeg_channels[axis] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[axis].append(value)
                
                # Process gyroscope data
                elif data_type == 'gyro' and len(args) == 3:
                    axes = ['gyro_x', 'gyro_y', 'gyro_z']
                    for axis, value in zip(axes, args):
                        if axis not in self.eeg_channels:
                            self.eeg_channels[axis] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[axis].append(value)
                
                # Process optical/fNIRS data
                elif data_type == 'optics':
                    for i, value in enumerate(args):
                        channel_name = f'optical_{i+1}'
                        if channel_name not in self.fnirs_channels:
                            self.fnirs_channels[channel_name] = deque(maxlen=self.buffer_size)
                        self.fnirs_channels[channel_name].append(value)
                
                # Process DRL/REF data
                elif data_type == 'drlref' and len(args) >= 2:
                    if 'DRL' not in self.eeg_channels:
                        self.eeg_channels['DRL'] = deque(maxlen=self.buffer_size)
                    if 'REF' not in self.eeg_channels:
                        self.eeg_channels['REF'] = deque(maxlen=self.buffer_size)
                    self.eeg_channels['DRL'].append(args[0])
                    self.eeg_channels['REF'].append(args[1])
                    
    def _print_detected_channels(self):
        """Print detected channels"""
        print("\nDetected channels:")
        print(f"EEG channels: {list(self.eeg_channels.keys())}")
        print(f"fNIRS channels: {list(self.fnirs_channels.keys())}")
        
    def get_latest_data(self):
        """Get the latest data from all channels"""
        with self.lock:
            eeg_data = {ch: list(data) for ch, data in self.eeg_channels.items()}
            fnirs_data = {ch: list(data) for ch, data in self.fnirs_channels.items()}
            timestamps = list(self.timestamps)
            
        return eeg_data, fnirs_data, timestamps
    
    def stop(self):
        """Stop the receiver"""
        self.running = False
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join()
        if self.socket:
            self.socket.close()
        print("Parser stopped")
    
    def setup_plots(self):
        """Configure all plot aesthetics and labels"""
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid layout - now with 5 rows to include FFT
        gs = GridSpec(5, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 1], 
                     width_ratios=[3, 1])
        
        # Main plots
        self.ax_eeg = self.fig.add_subplot(gs[0, 0])
        self.ax_fft = self.fig.add_subplot(gs[1, 0])  # NEW: FFT plot
        self.ax_fnirs = self.fig.add_subplot(gs[2, 0])
        self.ax_motion = self.fig.add_subplot(gs[3, 0])
        
        # Info panels
        self.ax_eeg_info = self.fig.add_subplot(gs[0, 1])
        self.ax_fft_info = self.fig.add_subplot(gs[1, 1])  # NEW: FFT info
        self.ax_fnirs_info = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[3, 1])
        
        # Status bar
        self.ax_status = self.fig.add_subplot(gs[4, :])
        
        # EEG plot setup
        self.ax_eeg.set_title('EEG Channels', fontsize=14, color='#4ECDC4', pad=10)
        self.ax_eeg.set_ylabel('Amplitude (μV)', fontsize=10)
        self.ax_eeg.set_xlabel('Time (s)', fontsize=10)
        self.ax_eeg.grid(True, alpha=0.2, linestyle='--')
        self.ax_eeg.set_facecolor('#1a1a1a')
        self.ax_eeg.set_ylim(0, 1700)  # Typical Muse range
        self.ax_eeg.set_xlim(-self.window_duration, 0)  # Initial time window
        
        # FFT plot setup
        self.ax_fft.set_title('EEG Frequency Spectrum', fontsize=14, color='#45B7D1', pad=10)
        self.ax_fft.set_ylabel('Intensity', fontsize=10)
        self.ax_fft.set_xlabel('Frequency (Hz)', fontsize=10)
        self.ax_fft.set_xlim(0, 70)
        self.ax_fft.set_ylim(0, 100)  # Initial limits
        self.ax_fft.grid(True, alpha=0.2, linestyle='--')
        self.ax_fft.set_facecolor('#1a1a1a')
        
        # fNIRS plot setup  
        self.ax_fnirs.set_title('fNIRS/Optical Channels', fontsize=14, color='#FF6B6B', pad=10)
        self.ax_fnirs.set_ylabel('Intensity (a.u.)', fontsize=10)
        self.ax_fnirs.set_xlabel('Time (s)', fontsize=10)
        self.ax_fnirs.grid(True, alpha=0.2, linestyle='--')
        self.ax_fnirs.set_facecolor('#1a1a1a')
        self.ax_fnirs.set_ylim(0, 1000)  # Initial limits
        self.ax_fnirs.set_xlim(-self.window_duration, 0)  # Initial time window
        
        # Motion plot setup
        self.ax_motion.set_title('Motion Sensors', fontsize=12, color='#96CEB4', pad=10)
        self.ax_motion.set_ylabel('Acc (g) / Gyro (°/s)', fontsize=10)
        self.ax_motion.set_xlabel('Time (s)', fontsize=10)
        self.ax_motion.grid(True, alpha=0.2, linestyle='--')
        self.ax_motion.set_facecolor('#1a1a1a')
        self.ax_motion.set_ylim(-2, 2)  # Initial limits
        self.ax_motion.set_xlim(-self.window_duration, 0)  # Initial time window
        
        # Info panels setup
        for ax in [self.ax_eeg_info, self.ax_fft_info, self.ax_fnirs_info, self.ax_stats]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#1a1a1a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        self.ax_eeg_info.set_title('EEG Stats', fontsize=12, color='#4ECDC4')
        self.ax_fft_info.set_title('Frequency Peaks', fontsize=12, color='#45B7D1')
        self.ax_fnirs_info.set_title('fNIRS Stats', fontsize=12, color='#FF6B6B')
        self.ax_stats.set_title('System Stats', fontsize=12, color='#FFD93D')
        
        # Status bar setup
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        self.ax_status.set_facecolor('#0a0a0a')
        for spine in self.ax_status.spines.values():
            spine.set_visible(False)
        
        # Layout
        plt.tight_layout()
    
    def initialize_plots(self):
        """Initialize plot lines based on detected channels"""
        # Wait for parser to detect data format
        wait_time = 0
        while not self.data_format_detected and wait_time < 10:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not self.data_format_detected:
            print("Warning: Starting visualization without detected channels")
            return
        
        print("Initializing visualization...")
        
        # Setup EEG lines (excluding motion sensors)
        eeg_channels = [ch for ch in self.eeg_channels.keys() 
                       if not ch.startswith('acc_') and not ch.startswith('gyro_')]
        
        for i, channel_name in enumerate(eeg_channels):
            color = self.eeg_colors.get(channel_name, '#FFFFFF')
            line, = self.ax_eeg.plot([], [], label=channel_name, color=color, 
                                    linewidth=1.5, alpha=0.9)
            self.eeg_lines[channel_name] = line
        
        # Setup FFT lines for main EEG channels
        for channel_name in ['TP9', 'AF7', 'AF8', 'TP10']:
            if channel_name in self.eeg_channels:
                color = self.eeg_colors.get(channel_name, '#FFFFFF')
                line, = self.ax_fft.plot([], [], label=channel_name, color=color, 
                                        linewidth=2, alpha=0.9)
                self.fft_lines[channel_name] = line
        
        # Setup fNIRS lines
        for i, channel_name in enumerate(sorted(self.fnirs_channels.keys())):
            color = self.fnirs_colors[i % len(self.fnirs_colors)]
            line, = self.ax_fnirs.plot([], [], label=channel_name, color=color, 
                                      linewidth=1.5, alpha=0.9)
            self.fnirs_lines[channel_name] = line
        
        # Setup motion sensor lines
        motion_channels = [ch for ch in self.eeg_channels.keys() 
                          if ch.startswith('acc_') or ch.startswith('gyro_')]
        
        for channel_name in motion_channels:
            color = self.motion_colors.get(channel_name, '#FFFFFF')
            line, = self.ax_motion.plot([], [], label=channel_name, color=color, 
                                       linewidth=1.5, alpha=0.9)
            self.motion_lines[channel_name] = line
        
        # Add legends
        if self.eeg_lines:
            self.ax_eeg.legend(loc='upper right', fontsize=8, framealpha=0.5)
        if self.fft_lines:
            self.ax_fft.legend(loc='upper right', fontsize=8, framealpha=0.5)
        if self.fnirs_lines:
            # Only show legend if not too many channels
            if len(self.fnirs_lines) <= 8:
                self.ax_fnirs.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
        if self.motion_lines:
            self.ax_motion.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
            
        print(f"Initialized {len(self.eeg_lines)} EEG, {len(self.fnirs_lines)} fNIRS, "
              f"and {len(self.motion_lines)} motion channels")
    
    def compute_fft(self, data, sampling_rate):
        """Compute FFT of the signal"""
        if len(data) < self.fft_window_samples:
            return None, None
        
        # Take the most recent samples
        signal_data = np.array(data[-self.fft_window_samples:])
        
        # Remove DC component
        signal_data = signal_data - np.mean(signal_data)
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(signal_data))
        signal_data = signal_data * window
        
        # Compute FFT
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(len(signal_data), 1/sampling_rate)
        
        # Only keep positive frequencies up to 70 Hz
        positive_freq_idx = (fft_freq > 0) & (fft_freq <= 70)
        
        return fft_freq[positive_freq_idx], np.abs(fft_vals[positive_freq_idx])
    
    def update_plots(self, frame):
        """Update all plots with latest data"""
        try:
            eeg_data, fnirs_data, timestamps = self.get_latest_data()
            
            if not timestamps:
                return list(self.eeg_lines.values()) + list(self.fnirs_lines.values()) + \
                       list(self.motion_lines.values()) + list(self.fft_lines.values())
            
            # Calculate time axis in seconds
            if len(timestamps) > 1:
                dt = np.diff(timestamps).mean()
                time_axis = np.arange(len(timestamps)) * dt
                time_axis = time_axis - time_axis[-1]  # Make most recent = 0
                # Debug output
                if frame % 50 == 0:
                    print(f"Time axis: {len(time_axis)} points, dt={dt:.3f}s, range=[{time_axis[0]:.1f}, {time_axis[-1]:.1f}]")
            else:
                time_axis = np.array([0])
                dt = 1.0  # Default
            
            # Determine display window
            display_samples = int(self.window_duration / dt) if len(timestamps) > 1 else 100
            
            # Update EEG plots
            eeg_values_for_scaling = []
            for channel_name, line in self.eeg_lines.items():
                if channel_name in eeg_data and len(eeg_data[channel_name]) > 0:
                    data = np.array(eeg_data[channel_name])
                    
                    # Don't apply high-pass filter initially - show raw data
                    # This helps with initial scaling
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    if n_samples > 0:
                        line.set_data(time_axis[-n_samples:], data[-n_samples:])
                        eeg_values_for_scaling.extend(data[-n_samples:])
            
            # Update FFT plots
            for channel_name, line in self.fft_lines.items():
                if channel_name in eeg_data and len(eeg_data[channel_name]) >= self.fft_window_samples:
                    freq, intensity = self.compute_fft(eeg_data[channel_name], self.sampling_rate)
                    if freq is not None:
                        line.set_data(freq, intensity)
            
            # Update fNIRS plots  
            fnirs_values_for_scaling = []
            for channel_name, line in self.fnirs_lines.items():
                if channel_name in fnirs_data and len(fnirs_data[channel_name]) > 0:
                    data = np.array(fnirs_data[channel_name])
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    if n_samples > 0:
                        line.set_data(time_axis[-n_samples:], data[-n_samples:])
                        fnirs_values_for_scaling.extend(data[-n_samples:])
            
            # Update motion plots
            motion_values_for_scaling = []
            for channel_name, line in self.motion_lines.items():
                if channel_name in eeg_data and len(eeg_data[channel_name]) > 0:
                    data = np.array(eeg_data[channel_name])
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    if n_samples > 0:
                        line.set_data(time_axis[-n_samples:], data[-n_samples:])
                        motion_values_for_scaling.extend(data[-n_samples:])
            
            # Update axis limits
            if len(time_axis) > 0:
                # Time axis
                for ax in [self.ax_eeg, self.ax_fnirs, self.ax_motion]:
                    ax.set_xlim(-self.window_duration, 0)
                
                # Y-axis scaling with immediate response
                # EEG Y-axis
                if eeg_values_for_scaling:
                    eeg_min = np.min(eeg_values_for_scaling)
                    eeg_max = np.max(eeg_values_for_scaling)
                    # If range is too small, use reasonable defaults for Muse
                    if abs(eeg_max - eeg_min) < 1:
                        eeg_center = (eeg_max + eeg_min) / 2
                        eeg_min = eeg_center - 50
                        eeg_max = eeg_center + 50
                    margin = (eeg_max - eeg_min) * 0.2
                    self.ax_eeg.set_ylim(eeg_min - margin, eeg_max + margin)
                    # Debug output every 50 frames
                    if frame % 50 == 0:
                        print(f"EEG range: {eeg_min:.2f} to {eeg_max:.2f} (n={len(eeg_values_for_scaling)} values)")
                else:
                    # Set typical Muse range if no data yet
                    self.ax_eeg.set_ylim(0, 1700)
                
                # fNIRS Y-axis
                if fnirs_values_for_scaling:
                    fnirs_min = np.min(fnirs_values_for_scaling)
                    fnirs_max = np.max(fnirs_values_for_scaling)
                    # If range is too small, use reasonable defaults
                    if abs(fnirs_max - fnirs_min) < 1:
                        fnirs_center = (fnirs_max + fnirs_min) / 2
                        fnirs_min = fnirs_center - 50
                        fnirs_max = fnirs_center + 50
                    margin = (fnirs_max - fnirs_min) * 0.2
                    self.ax_fnirs.set_ylim(fnirs_min - margin, fnirs_max + margin)
                    # Debug output every 50 frames
                    if frame % 50 == 0:
                        print(f"fNIRS range: {fnirs_min:.2f} to {fnirs_max:.2f} (n={len(fnirs_values_for_scaling)} values)")
                else:
                    # Set default range if no data yet
                    self.ax_fnirs.set_ylim(0, 1000)
                
                # Motion Y-axis
                if motion_values_for_scaling:
                    motion_min = np.min(motion_values_for_scaling)
                    motion_max = np.max(motion_values_for_scaling)
                    margin = (motion_max - motion_min) * 0.2 if motion_max != motion_min else 1
                    self.ax_motion.set_ylim(motion_min - margin, motion_max + margin)
                else:
                    # Set default range if no data yet
                    self.ax_motion.set_ylim(-2, 2)
                
                # FFT Y-axis auto-scaling
                self.ax_fft.relim()
                self.ax_fft.autoscale_view(scaley=True, scalex=False)
            
            # Update info panels
            self.update_info_panels(eeg_data, fnirs_data, timestamps)
            
            # Update sample rate
            current_time = time.time()
            self.sample_times.append(current_time)
            if len(self.sample_times) > 1:
                rate = len(self.sample_times) / (self.sample_times[-1] - self.sample_times[0])
                self.update_status_bar(rate, len(timestamps))
            
            # Force redraw of all axes
            self.ax_eeg.figure.canvas.draw_idle()
            self.ax_fft.figure.canvas.draw_idle()
            self.ax_fnirs.figure.canvas.draw_idle()
            self.ax_motion.figure.canvas.draw_idle()
        
        except Exception as e:
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
        
        return list(self.eeg_lines.values()) + list(self.fnirs_lines.values()) + \
               list(self.motion_lines.values()) + list(self.fft_lines.values())
    
    def update_info_panels(self, eeg_data, fnirs_data, timestamps):
        """Update the information panels with statistics"""
        # Clear previous text
        self.ax_eeg_info.clear()
        self.ax_fft_info.clear()
        self.ax_fnirs_info.clear()
        self.ax_stats.clear()
        
        # Reconfigure after clearing
        for ax in [self.ax_eeg_info, self.ax_fft_info, self.ax_fnirs_info, self.ax_stats]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # EEG statistics
        y_pos = 0.9
        self.ax_eeg_info.text(0.1, y_pos, 'Channel Stats:', fontsize=10, 
                             weight='bold', color='#4ECDC4', transform=self.ax_eeg_info.transAxes)
        y_pos -= 0.15
        
        for channel in ['TP9', 'AF7', 'AF8', 'TP10']:
            if channel in eeg_data and len(eeg_data[channel]) > 0:
                data = np.array(eeg_data[channel][-100:])  # Last 100 samples
                mean_val = np.mean(data)
                std_val = np.std(data)
                color = self.eeg_colors.get(channel, '#FFFFFF')
                self.ax_eeg_info.text(0.1, y_pos, f'{channel}:', fontsize=9, 
                                     color=color, transform=self.ax_eeg_info.transAxes)
                self.ax_eeg_info.text(0.5, y_pos, f'{mean_val:.1f}±{std_val:.1f} μV', 
                                     fontsize=9, color='white', 
                                     transform=self.ax_eeg_info.transAxes)
                y_pos -= 0.12
        
        # FFT frequency peaks info
        y_pos = 0.9
        self.ax_fft_info.text(0.1, y_pos, 'Dominant Freqs:', fontsize=10, 
                             weight='bold', color='#45B7D1', transform=self.ax_fft_info.transAxes)
        y_pos -= 0.15
        
        for channel_name in ['TP9', 'AF7', 'AF8', 'TP10']:
            if channel_name in eeg_data and len(eeg_data[channel_name]) >= self.fft_window_samples:
                freq, intensity = self.compute_fft(eeg_data[channel_name], self.sampling_rate)
                if freq is not None and len(intensity) > 0:
                    # Find dominant frequency
                    peak_idx = np.argmax(intensity)
                    peak_freq = freq[peak_idx]
                    color = self.eeg_colors.get(channel_name, '#FFFFFF')
                    self.ax_fft_info.text(0.1, y_pos, f'{channel_name}:', fontsize=9, 
                                         color=color, transform=self.ax_fft_info.transAxes)
                    self.ax_fft_info.text(0.5, y_pos, f'{peak_freq:.1f} Hz', 
                                         fontsize=9, color='white', 
                                         transform=self.ax_fft_info.transAxes)
                    y_pos -= 0.12
        
        # fNIRS statistics
        y_pos = 0.9
        self.ax_fnirs_info.text(0.1, y_pos, 'Optical Stats:', fontsize=10, 
                               weight='bold', color='#FF6B6B', 
                               transform=self.ax_fnirs_info.transAxes)
        y_pos -= 0.15
        
        # Group optical channels
        if fnirs_data:
            all_values = []
            for ch in fnirs_data.values():
                if len(ch) > 0:
                    all_values.extend(ch[-100:])
            
            if all_values:
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                min_val = np.min(all_values)
                max_val = np.max(all_values)
                
                self.ax_fnirs_info.text(0.1, y_pos, 'Mean:', fontsize=9, 
                                       color='white', transform=self.ax_fnirs_info.transAxes)
                self.ax_fnirs_info.text(0.5, y_pos, f'{mean_val:.2f}±{std_val:.2f}', 
                                       fontsize=9, color='white', 
                                       transform=self.ax_fnirs_info.transAxes)
                y_pos -= 0.12
                
                self.ax_fnirs_info.text(0.1, y_pos, 'Range:', fontsize=9, 
                                       color='white', transform=self.ax_fnirs_info.transAxes)
                self.ax_fnirs_info.text(0.5, y_pos, f'{min_val:.2f} - {max_val:.2f}', 
                                       fontsize=9, color='white', 
                                       transform=self.ax_fnirs_info.transAxes)
        
        # System statistics
        y_pos = 0.9
        self.ax_stats.text(0.1, y_pos, 'System Info:', fontsize=10, 
                          weight='bold', color='#FFD93D', transform=self.ax_stats.transAxes)
        y_pos -= 0.2
        
        self.ax_stats.text(0.1, y_pos, 'Buffer:', fontsize=9, 
                          color='white', transform=self.ax_stats.transAxes)
        self.ax_stats.text(0.5, y_pos, f'{len(timestamps)} samples', fontsize=9, 
                          color='white', transform=self.ax_stats.transAxes)
        y_pos -= 0.15
        
        if len(timestamps) > 100:
            # Calculate actual sample rate
            time_diff = timestamps[-1] - timestamps[-100]
            actual_rate = 99 / time_diff if time_diff > 0 else 0
            self.ax_stats.text(0.1, y_pos, 'Rate:', fontsize=9, 
                              color='white', transform=self.ax_stats.transAxes)
            self.ax_stats.text(0.5, y_pos, f'{actual_rate:.1f} Hz', fontsize=9, 
                              color='white', transform=self.ax_stats.transAxes)
    
    def update_status_bar(self, display_rate, buffer_size):
        """Update the status bar with system information"""
        self.ax_status.clear()
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        for spine in self.ax_status.spines.values():
            spine.set_visible(False)
        
        status_text = (f'Display Rate: {display_rate:.1f} Hz | '
                      f'Buffer: {buffer_size} samples | '
                      f'Window: {self.window_duration}s | '
                      f'Port: {self.port} | '
                      f'FFT Window: {self.fft_window_samples} samples')
        
        self.ax_status.text(0.5, 0.5, status_text, fontsize=10, 
                           ha='center', va='center', color='#888888',
                           transform=self.ax_status.transAxes)
    
    def start(self):
        """Start real-time visualization"""
        print("Starting real-time visualization...")
        
        # Setup plots first
        self.setup_plots()
        
        # Start parser
        if not self.start_receiving():
            print("Failed to start data parser")
            return
        
        # Initialize plots
        self.initialize_plots()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plots, 
            interval=self.update_interval, 
            blit=False, 
            cache_frame_data=False
        )
        
        # Add keyboard shortcuts
        def on_key(event):
            if event.key == 'q':
                plt.close('all')
            elif event.key == '+' or event.key == '=':
                self.window_duration = min(self.window_duration + 2, 30)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == '-':
                self.window_duration = max(self.window_duration - 2, 2)
                print(f"Window duration: {self.window_duration}s")
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("\nVisualization started!")
        print("Keyboard shortcuts:")
        print("  '+' / '-' : Increase/decrease time window")
        print("  'q' : Quit")
        print("\nClose window to stop.")
        
        plt.show()
        
        # Cleanup
        self.stop()

if __name__ == "__main__":
    # Create standalone visualizer
    visualizer = EEGfNIRSVisualizerStandalone(
        port=8052, 
        buffer_size=2000,
        update_interval=50, 
        window_duration=10
    )
    
    print("=" * 60)
    print("EEG/fNIRS Real-Time Visualizer with Frequency Analysis")
    print("=" * 60)
    print(f"Waiting for UDP data on port {visualizer.port}...")
    print("Make sure your Muse S Athena is streaming to this port.")
    print("")
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        visualizer.stop()