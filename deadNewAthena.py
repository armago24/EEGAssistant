#!/usr/bin/env python3
"""
Visualizer for Muse S Athena with EEG and fNIRS (optics) data
Adapted for the specific Athena packet format
"""

import socket
import struct
import threading
import time
import numpy as np
from collections import deque
from scipy import signal
import matplotlib
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for better rendering
except:
    pass  # Fall back to default backend if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import json
import os
from datetime import datetime
from tkinter import filedialog, messagebox
import tkinter as tk

class MuseAthenaVisualizer:
    def __init__(self, port=8052, buffer_size=2000, window_duration=10):
        # Network settings
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = buffer_size
        self.window_duration = window_duration
        self.timestamps = deque(maxlen=buffer_size)
        
        # Channel data storage - 4 EEG channels for Athena
        self.eeg_channels = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size),
            'AF8': deque(maxlen=buffer_size),
            'TP10': deque(maxlen=buffer_size)
        }
        
        # fNIRS/Optics channels - 8 values
        # First 4 appear to be normalized intensities, last 4 raw readings
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
        
        # Visualization
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.spectral_lines = {}
        
        # Color schemes
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
        
        # Spectral analysis parameters
        self.sample_rate = 256  # Muse EEG sample rate
        self.spectral_window_size = 512  # FFT window size
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        self.max_freq = 70  # Maximum frequency to display
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Recording functionality
        self.is_recording = False
        self.recording_start_time = None
        self.recorded_data = {
            'eeg': [],
            'fnirs': [],
            'motion': [],
            'ref': [],
            'events': []
        }
        self.record_button = None
        
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
        type_tags = message['type_tags']
        timestamp = time.time()
        
        with self.lock:
            # Parse address: /username/datatype
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                # EEG data - 4 floats for Athena
                if data_type == 'eeg' and len(args) == 4:
                    # Add timestamp
                    self.timestamps.append(timestamp)
                    
                    channels = ['TP9', 'AF7', 'AF8', 'TP10']
                    for ch, val in zip(channels, args):
                        self.eeg_channels[ch].append(val)
                    self.eeg_packet_count += 1
                    
                    # Record data if recording
                    if self.is_recording:
                        self.recorded_data['eeg'].append({
                            'timestamp': timestamp,
                            'TP9': args[0],
                            'AF7': args[1],
                            'AF8': args[2],
                            'TP10': args[3]
                        })
                    
                    # Debug first few packets
                    if self.eeg_packet_count <= 5:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                # fNIRS/Optics data - 8 floats
                elif data_type == 'optics' and len(args) == 8:
                    # First 4 values appear to be normalized (0-1 range)
                    # Last 4 values appear to be raw sensor readings
                    norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
                    raw_channels = ['Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw']
                    
                    for i, (ch, val) in enumerate(zip(norm_channels + raw_channels, args)):
                        self.fnirs_channels[ch].append(val)
                    
                    self.fnirs_packet_count += 1
                    
                    # Record data if recording
                    if self.is_recording:
                        self.recorded_data['fnirs'].append({
                            'timestamp': timestamp,
                            'normalized': args[:4],
                            'raw': args[4:]
                        })
                    
                    # Debug first few packets
                    if self.fnirs_packet_count <= 5:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}, raw={args[4:]}")
                
                # Accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    channels = ['acc_x', 'acc_y', 'acc_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    # Record data if recording
                    if self.is_recording:
                        self.recorded_data['motion'].append({
                            'timestamp': timestamp,
                            'type': 'acc',
                            'x': args[0],
                            'y': args[1],
                            'z': args[2]
                        })
                
                # Gyroscope data
                elif data_type == 'gyro' and len(args) == 3:
                    channels = ['gyro_x', 'gyro_y', 'gyro_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    # Record data if recording
                    if self.is_recording:
                        self.recorded_data['motion'].append({
                            'timestamp': timestamp,
                            'type': 'gyro',
                            'x': args[0],
                            'y': args[1],
                            'z': args[2]
                        })
                
                # DRL/REF data
                elif data_type == 'drlref' and len(args) >= 2:
                    self.ref_channels['DRL'].append(args[0])
                    self.ref_channels['REF'].append(args[1])
                    
                    # Record data if recording
                    if self.is_recording:
                        self.recorded_data['ref'].append({
                            'timestamp': timestamp,
                            'DRL': args[0],
                            'REF': args[1]
                        })
    
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
    
    def setup_visualization(self):
        """Setup matplotlib figure and axes"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid: 6 rows for different data types
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
        self.axes['eeg'].set_ylabel('Amplitude (μV)')
        
        self.axes['spectral'].set_title('Spectral Analysis - Power Spectral Density', fontsize=14, color='#FFD93D', pad=10)
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        
        self.axes['fnirs'].set_title('fNIRS/Optics - Functional Near-Infrared Spectroscopy', fontsize=14, color='#E74C3C', pad=10)
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
        self.axes['info'].set_title('Signal Statistics', fontsize=14, color='#FFD93D', pad=10)
        
        # Add record button
        ax_button = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.record_button = Button(ax_button, 'Start Recording', 
                                   color='#2a2a2a', hovercolor='#3a3a3a')
        self.record_button.on_clicked(self.toggle_recording)
        
        # Initialize plot lines
        self._initialize_lines()
        
        plt.tight_layout()
    
    def toggle_recording(self, event):
        """Toggle recording state"""
        with self.lock:
            if not self.is_recording:
                # Start recording
                self.is_recording = True
                self.recording_start_time = time.time()
                self.recorded_data = {
                    'eeg': [],
                    'fnirs': [],
                    'motion': [],
                    'ref': [],
                    'events': [],
                    'metadata': {
                        'start_time': self.recording_start_time,
                        'sample_rate': self.sample_rate,
                        'device': 'Muse S Athena'
                    }
                }
                self.record_button.label.set_text('Stop Recording')
                self.record_button.color = '#ff4444'
                self.record_button.hovercolor = '#ff6666'
                print(f"\nRecording started at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("Press 'c' for confused, 'o' for overwhelmed, 'd' for dictionary")
            else:
                # Stop recording
                self.is_recording = False
                self.record_button.label.set_text('Start Recording')
                self.record_button.color = '#2a2a2a'
                self.record_button.hovercolor = '#3a3a3a'
                print(f"\nRecording stopped. Duration: {time.time() - self.recording_start_time:.1f}s")
                print(f"Events recorded: {len(self.recorded_data['events'])}")
    
    def record_event(self, event_type):
        """Record an event with timestamp"""
        if self.is_recording:
            timestamp = time.time()
            event = {
                'timestamp': timestamp,
                'type': event_type,
                'relative_time': timestamp - self.recording_start_time
            }
            self.recorded_data['events'].append(event)
            print(f"Event recorded: {event_type} at {event['relative_time']:.2f}s")
    
    def save_recording(self):
        """Save the recorded data to a file"""
        if not self.recorded_data['eeg'] and not self.recorded_data['events']:
            return
        
        # Create Tkinter root window (hidden)
        root = tk.Tk()
        root.withdraw()
        
        # Ask for filename
        default_name = f"muse_athena_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = filedialog.asksaveasfilename(
            initialdir=os.path.expanduser("~/Downloads"),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            # Update metadata
            self.recorded_data['metadata']['end_time'] = time.time()
            self.recorded_data['metadata']['duration'] = (
                self.recorded_data['metadata']['end_time'] - 
                self.recorded_data['metadata']['start_time']
            )
            self.recorded_data['metadata']['total_eeg_samples'] = len(self.recorded_data['eeg'])
            self.recorded_data['metadata']['total_fnirs_samples'] = len(self.recorded_data['fnirs'])
            self.recorded_data['metadata']['total_events'] = len(self.recorded_data['events'])
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(self.recorded_data, f, indent=2)
            
            print(f"\nRecording saved to: {filename}")
            print(f"Total EEG samples: {len(self.recorded_data['eeg'])}")
            print(f"Total fNIRS samples: {len(self.recorded_data['fnirs'])}")
            print(f"Total events: {len(self.recorded_data['events'])}")
            
            # Show summary of events
            event_counts = {}
            for event in self.recorded_data['events']:
                event_type = event['type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event_counts:
                print("\nEvent summary:")
                for event_type, count in event_counts.items():
                    print(f"  {event_type}: {count}")
        
        root.destroy()
    
    def _initialize_lines(self):
        """Initialize all plot lines"""
        # EEG lines - 4 channels
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95,
                                         linestyle='-', marker='',
                                         antialiased=True)
            self.lines[f'eeg_{ch}'] = line
        
        # Initialize spectral plot
        self.spectral_lines = {}
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color, 
                                              linewidth=1.5, alpha=0.9,
                                              antialiased=True)
            self.spectral_lines[ch] = line
        
        # Set up spectral axis
        self.axes['spectral'].set_xlim(0, self.max_freq)
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        
        # Add frequency band labels
        for band_name, (low, high) in self.freq_bands.items():
            self.axes['spectral'].axvline(x=low, color='white', linestyle=':', alpha=0.3, linewidth=0.5)
            mid_freq = (low + high) / 2
            if mid_freq < self.max_freq:
                self.axes['spectral'].text(mid_freq, 0.98, band_name, 
                                         fontsize=8, color='white', alpha=0.7,
                                         horizontalalignment='center',
                                         verticalalignment='top',
                                         transform=self.axes['spectral'].get_xaxis_transform())
        
        # fNIRS lines - normalized values only for main display
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color,
                                          linewidth=1.5, alpha=0.9,
                                          antialiased=True)
            self.lines[f'fnirs_{ch_norm}'] = line
        
        # Motion lines - accelerometer
        acc_colors = ['#3498DB', '#2ECC71', '#9B59B6']
        for i, ch in enumerate(['acc_x', 'acc_y', 'acc_z']):
            line, = self.axes['motion'].plot([], [], label=ch.replace('acc_', ''),
                                           color=acc_colors[i],
                                           linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Gyroscope lines
        gyro_colors = ['#F39C12', '#E67E22', '#D35400']
        for i, ch in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            line, = self.axes['gyro'].plot([], [], label=ch.replace('gyro_', ''),
                                         color=gyro_colors[i],
                                         linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Reference lines
        ref_colors = ['#95A5A6', '#7F8C8D']
        for i, ch in enumerate(['DRL', 'REF']):
            line, = self.axes['ref'].plot([], [], label=ch,
                                        color=ref_colors[i],
                                        linewidth=1.5, alpha=0.9)
            self.lines[f'ref_{ch}'] = line
        
        # Add legends
        self.axes['eeg'].legend(loc='upper left', fontsize=8, ncol=4, 
                               framealpha=0.7, bbox_to_anchor=(0.3, 1))
        self.axes['spectral'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        self.axes['fnirs'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.5)
        self.axes['motion'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['gyro'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['ref'].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
    
    def compute_spectrum(self, data, sample_rate=256):
        """Compute power spectral density using Welch's method"""
        if len(data) < self.spectral_window_size:
            return None, None
        
        # Use Welch's method for more stable spectrum estimation
        frequencies, psd = signal.welch(
            data, 
            fs=sample_rate, 
            nperseg=min(len(data), self.spectral_window_size),
            noverlap=min(len(data)//2, self.spectral_window_size//2),
            scaling='density'
        )
        
        # Limit to 0-70 Hz
        freq_mask = frequencies <= self.max_freq
        frequencies = frequencies[freq_mask]
        psd = psd[freq_mask]
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        return frequencies, psd_db
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Get the minimum length across all EEG channels to ensure alignment
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Calculate time axis based on actual timestamps
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                # Use actual time differences
                time_axis = timestamps - timestamps[-1]  # Make most recent = 0
                
                # Calculate display window based on time
                display_mask = time_axis >= -self.window_duration
                display_samples = np.sum(display_mask)
            else:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Update EEG
            eeg_values_for_scaling = []
            filtered_eeg_data = {}
            
            for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        # Get aligned data
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
                        # Apply high-pass filter to remove DC
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
                        
                        # Store for spectral analysis
                        filtered_eeg_data[ch_name] = filtered_data
                        
                        # Apply display mask
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            # Update spectral analysis
            if len(filtered_eeg_data) == 4:
                all_psd_values = []
                
                for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                    if ch_name in filtered_eeg_data and ch_name in self.spectral_lines:
                        spectral_window_samples = min(len(filtered_eeg_data[ch_name]), 
                                                    int(self.sample_rate * 4))
                        data_for_spectrum = filtered_eeg_data[ch_name][-spectral_window_samples:]
                        
                        frequencies, psd = self.compute_spectrum(data_for_spectrum, self.sample_rate)
                        
                        if frequencies is not None and psd is not None:
                            self.spectral_lines[ch_name].set_data(frequencies, psd)
                            all_psd_values.extend(psd)
                
                # Auto-scale y-axis
                if all_psd_values:
                    y_min = np.percentile(all_psd_values, 5) - 5
                    y_max = np.percentile(all_psd_values, 95) + 5
                    self.axes['spectral'].set_ylim(y_min, y_max)
            
            # Update fNIRS - show normalized values
            fnirs_norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
            for ch_name in fnirs_norm_channels:
                if ch_name in self.fnirs_channels and len(self.fnirs_channels[ch_name]) > 0:
                    line_key = f'fnirs_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.fnirs_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Motion (accelerometer)
            for ch_name in ['acc_x', 'acc_y', 'acc_z']:
                if ch_name in self.motion_channels and len(self.motion_channels[ch_name]) > 0:
                    line_key = f'motion_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.motion_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Gyroscope
            for ch_name in ['gyro_x', 'gyro_y', 'gyro_z']:
                if ch_name in self.motion_channels and len(self.motion_channels[ch_name]) > 0:
                    line_key = f'motion_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.motion_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Reference
            for ch_name, data in self.ref_channels.items():
                if len(data) > 0:
                    line_key = f'ref_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(data))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update axes limits
            for ax in [self.axes['eeg'], self.axes['fnirs'], 
                      self.axes['motion'], self.axes['gyro'], self.axes['ref']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            # Special handling for EEG scaling
            if eeg_values_for_scaling:
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                y_range = 4 * eeg_std
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
                
                self.axes['eeg'].text(0.02, 0.98, 
                                    f'Scale: ±{y_range/2:.1f} μV', 
                                    transform=self.axes['eeg'].transAxes, 
                                    fontsize=9,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Update info panel
            self._update_info_panel()
        
        return list(self.lines.values()) + list(self.spectral_lines.values())
    
    def _update_info_panel(self):
        """Update statistics panel"""
        self.axes['info'].clear()
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        y_pos = 0.95
        self.axes['info'].text(0.1, y_pos, 'Signal Statistics', 
                             fontsize=12, weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
        # Recording status
        if self.is_recording:
            y_pos -= 0.06
            elapsed = time.time() - self.recording_start_time
            self.axes['info'].text(0.1, y_pos, f'Recording: {elapsed:.1f}s', 
                                 fontsize=10, color='#ff4444', weight='bold',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'Events: {len(self.recorded_data["events"])}', 
                                 fontsize=9, color='#ff6666',
                                 transform=self.axes['info'].transAxes)
        
        # EEG stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'EEG (4ch):', fontsize=10, 
                             weight='bold', color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if ch in self.eeg_channels and len(self.eeg_channels[ch]) > 0:
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
        
        # Frequency band power
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, 'Band Power:', fontsize=10,
                             weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Calculate average band powers
        if hasattr(self, 'spectral_lines'):
            band_powers = {band: [] for band in self.freq_bands.keys()}
            
            for ch_name, line in self.spectral_lines.items():
                xdata, ydata = line.get_data()
                if len(xdata) > 0 and len(ydata) > 0:
                    for band_name, (low, high) in self.freq_bands.items():
                        band_mask = (xdata >= low) & (xdata <= high)
                        if np.any(band_mask):
                            band_power = np.mean(ydata[band_mask])
                            band_powers[band_name].append(band_power)
            
            for band_name in self.freq_bands.keys():
                if band_powers[band_name]:
                    avg_power = np.mean(band_powers[band_name])
                    self.axes['info'].text(0.15, y_pos, f'{band_name}:', fontsize=9,
                                         color='white',
                                         transform=self.axes['info'].transAxes)
                    self.axes['info'].text(0.4, y_pos, f'{avg_power:.1f} dB',
                                         fontsize=9, color='white',
                                         transform=self.axes['info'].transAxes)
                    y_pos -= 0.04
        
        # fNIRS stats
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, 'fNIRS:', fontsize=10,
                             weight='bold', color='#E74C3C',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Show both normalized and raw values
        for i, ch_base in enumerate(['Ch1', 'Ch2', 'Ch3', 'Ch4']):
            ch_norm = f'{ch_base}_norm'
            ch_raw = f'{ch_base}_raw'
            
            if ch_norm in self.fnirs_channels and len(self.fnirs_channels[ch_norm]) > 0:
                norm_data = np.array(list(self.fnirs_channels[ch_norm])[-100:])
                raw_data = np.array(list(self.fnirs_channels[ch_raw])[-100:])
                
                self.axes['info'].text(0.15, y_pos, f'{ch_base}:', fontsize=9,
                                     color=self.fnirs_colors[ch_base],
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.3, y_pos, f'{np.mean(norm_data):.3f}',
                                     fontsize=8, color='white',
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.5, y_pos, f'({np.mean(raw_data):.1f})',
                                     fontsize=8, color='#aaaaaa',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.035
        
        # System stats
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, 'System:', fontsize=10,
                             weight='bold', color='#95A5A6',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        self.axes['info'].text(0.15, y_pos, 'Packets:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.packet_count}',
                             fontsize=9, color='white',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.15, y_pos, 'EEG:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.eeg_packet_count}',
                             fontsize=9, color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.15, y_pos, 'fNIRS:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.fnirs_packet_count}',
                             fontsize=9, color='#E74C3C',
                             transform=self.axes['info'].transAxes)
    
    def start(self):
        """Start the visualizer"""
        print("Starting Muse S Athena Visualizer with EEG and fNIRS...")
        print(f"Listening for OSC data on UDP port {self.port}")
        print("\nDevice: Muse S Athena")
        print("\nExpected data format:")
        print("  - EEG: 4 channels (TP9, AF7, AF8, TP10)")
        print("  - fNIRS/Optics: 8 values (4 normalized + 4 raw)")
        print("  - Accelerometer: 3 axes")
        print("  - Gyroscope: 3 axes")
        print("  - Reference: DRL and REF")
        print("\nfNIRS Information:")
        print("  - Functional Near-Infrared Spectroscopy")
        print("  - Measures hemodynamic responses (blood oxygen levels)")
        print("  - Normalized values: 0-1 range")
        print("  - Raw values: sensor readings")
        print("\nSpectral Analysis shows frequency content of EEG signals (0-70 Hz):")
        print("  - Delta (0.5-4 Hz): Deep sleep")
        print("  - Theta (4-8 Hz): Drowsiness, meditation")
        print("  - Alpha (8-13 Hz): Relaxed, eyes closed")
        print("  - Beta (13-30 Hz): Active thinking, focus")
        print("  - Gamma (30-50 Hz): High-level cognitive processing")
        
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
            print(f"Failed to start receiver: {e}")
            return
        
        # Setup visualization
        self.setup_visualization()
        
        # Keyboard shortcuts
        def on_key(event):
            if event.key == 'q':
                plt.close('all')
                self.stop()
            elif event.key == '+' or event.key == '=':
                self.window_duration = min(self.window_duration + 2, 30)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == '-':
                self.window_duration = max(self.window_duration - 2, 2)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == 'r':
                # Reset buffers
                with self.lock:
                    for ch in self.eeg_channels.values():
                        ch.clear()
                    for ch in self.fnirs_channels.values():
                        ch.clear()
                    for ch in self.motion_channels.values():
                        ch.clear()
                    for ch in self.ref_channels.values():
                        ch.clear()
                    self.timestamps.clear()
                print("Buffers reset")
            elif event.key == 'c':
                self.record_event('confused')
            elif event.key == 'o':
                self.record_event('overwhelmed')
            elif event.key == 'd':
                self.record_event('dictionary')
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Handle window close event
        def on_close(event):
            if self.is_recording and (self.recorded_data['eeg'] or self.recorded_data['events']):
                # Stop recording if still active
                self.is_recording = False
                
                # Ask if user wants to save
                root = tk.Tk()
                root.withdraw()
                result = messagebox.askyesno("Save Recording?", 
                                           "Do you want to save the recording?")
                root.destroy()
                
                if result:
                    self.save_recording()
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=40,  # 25 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("\nVisualization started!")
        print("First 5 EEG and fNIRS packets will be printed for verification.")
        print("\nKeyboard shortcuts:")
        print("  '+'/'-' : Increase/decrease time window")
        print("  'r'     : Reset buffers")
        print("  'c'     : Mark confused event")
        print("  'o'     : Mark overwhelmed event")
        print("  'd'     : Mark dictionary event")
        print("  'q'     : Quit")
        print("\nClick 'Start Recording' button to begin recording session")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if self.socket:
            self.socket.close()
        print(f"\nVisualizer stopped")
        print(f"Total packets received: {self.packet_count}")
        print(f"EEG packets received: {self.eeg_packet_count}")
        print(f"fNIRS packets received: {self.fnirs_packet_count}")


if __name__ == "__main__":
    # Create and start visualizer
    visualizer = MuseAthenaVisualizer(
        port=8052,
        buffer_size=2000,
        window_duration=10
    )
    
    try:
        visualizer.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()