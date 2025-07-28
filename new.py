#!/usr/bin/env python3
"""
Fixed visualizer for your Muse device
Handles the specific packet format: no leading /, 6 EEG values with 2 NaN, etc.
"""

import socket
import struct
import threading
import time
import numpy as np
from collections import deque
import matplotlib
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for better rendering
except:
    pass  # Fall back to default backend if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

class MuseFixedVisualizer:
    def __init__(self, port=8052, buffer_size=2000, window_duration=10):
        # Network settings
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = buffer_size
        self.window_duration = window_duration
        self.timestamps = deque(maxlen=buffer_size)
        
        # Channel data storage - only 4 EEG channels since last 2 are NaN
        self.eeg_channels = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size),
            'AF8': deque(maxlen=buffer_size),
            'TP10': deque(maxlen=buffer_size)
        }
        
        # Only middle PPG channel seems valid
        self.ppg_channels = {
            'PPG': deque(maxlen=buffer_size)
        }
        
        self.motion_channels = {
            'acc_x': deque(maxlen=buffer_size),
            'acc_y': deque(maxlen=buffer_size),
            'acc_z': deque(maxlen=buffer_size)
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
        
        # Color schemes
        self.eeg_colors = {
            'TP9': '#FF6B6B',
            'AF7': '#4ECDC4',
            'AF8': '#45B7D1',
            'TP10': '#96CEB4'
        }
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.show_filtered = True  # Toggle for filtered vs raw EEG
        
    def parse_osc_string(self, data, offset):
        """Parse null-terminated, 4-byte aligned string from OSC data"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        offset = ((end + 4) // 4) * 4
        return string, offset
    
    def parse_osc_message(self, data):
        """Parse an OSC message - handles format without leading /"""
        try:
            offset = 0
            
            # Parse address (no leading / in your format)
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
                    # Skip NaN values
                    if not np.isnan(value):
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
                
                # EEG data - expects 6 floats but last 2 are NaN
                if data_type == 'eeg' and len(type_tags) == 6:
                    # We already filtered out NaN values, so args should have 4 values
                    if len(args) == 4:
                        # Add timestamp only when we have complete EEG data
                        self.timestamps.append(timestamp)
                        
                        channels = ['TP9', 'AF7', 'AF8', 'TP10']
                        for ch, val in zip(channels, args):
                            self.eeg_channels[ch].append(val)
                        self.eeg_packet_count += 1
                        
                        # Debug first few packets
                        if self.eeg_packet_count <= 5:
                            print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                # PPG data - only middle value is valid
                elif data_type == 'ppg' and len(args) >= 1:
                    # args should have filtered out the NaN values
                    if len(args) > 0:
                        self.ppg_channels['PPG'].append(args[0])
                
                # Accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    channels = ['acc_x', 'acc_y', 'acc_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                
                # DRL/REF data
                elif data_type == 'drlref' and len(args) >= 2:
                    self.ref_channels['DRL'].append(args[0])
                    self.ref_channels['REF'].append(args[1])
    
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
        
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid: 4 rows for different data types
        gs = GridSpec(4, 2, figure=self.fig, 
                     height_ratios=[3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        # Main plots
        self.axes['eeg'] = self.fig.add_subplot(gs[0, 0])
        self.axes['ppg'] = self.fig.add_subplot(gs[1, 0])
        self.axes['motion'] = self.fig.add_subplot(gs[2, 0])
        self.axes['ref'] = self.fig.add_subplot(gs[3, 0])
        
        # Info panel
        self.axes['info'] = self.fig.add_subplot(gs[:3, 1])
        
        # Configure axes
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name != 'info':
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        
        # Titles and labels
        self.axes['eeg'].set_title('EEG Channels (4 active channels)', fontsize=14, color='#4ECDC4', pad=10)
        self.axes['eeg'].set_ylabel('Amplitude (μV)')
        
        self.axes['ppg'].set_title('PPG (Photoplethysmography)', fontsize=14, color='#FF6B6B', pad=10)
        self.axes['ppg'].set_ylabel('Intensity')
        
        self.axes['motion'].set_title('Accelerometer', fontsize=14, color='#96CEB4', pad=10)
        self.axes['motion'].set_ylabel('Acceleration (g)')
        
        self.axes['ref'].set_title('Reference Electrodes', fontsize=12, color='#95A5A6', pad=10)
        self.axes['ref'].set_ylabel('Voltage')
        self.axes['ref'].set_xlabel('Time (s)')
        
        # Info panel setup
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        self.axes['info'].set_title('Signal Statistics', fontsize=14, color='#FFD93D', pad=10)
        
        # Initialize plot lines
        self._initialize_lines()
        
        plt.tight_layout()
    
    def _initialize_lines(self):
        """Initialize all plot lines"""
        # EEG lines - only 4 channels with smooth lines
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95,
                                         linestyle='-', marker='',
                                         antialiased=True)  # Smooth lines with anti-aliasing
            self.lines[f'eeg_{ch}'] = line
        
        # Single PPG line
        line, = self.axes['ppg'].plot([], [], label='PPG', 
                                     color='#E74C3C',
                                     linewidth=1.5, alpha=0.9)
        self.lines['ppg_PPG'] = line
        
        # Motion lines
        acc_colors = ['#3498DB', '#2ECC71', '#9B59B6']
        for i, ch in enumerate(['acc_x', 'acc_y', 'acc_z']):
            line, = self.axes['motion'].plot([], [], label=ch.replace('acc_', 'Axis '),
                                           color=acc_colors[i],
                                           linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Reference lines
        ref_colors = ['#95A5A6', '#7F8C8D']
        for i, ch in enumerate(['DRL', 'REF']):
            line, = self.axes['ref'].plot([], [], label=ch,
                                        color=ref_colors[i],
                                        linewidth=1.5, alpha=0.9)
            self.lines[f'ref_{ch}'] = line
        
        # Add legends with better positioning
        self.axes['eeg'].legend(loc='upper left', fontsize=8, ncol=4, 
                               framealpha=0.7, bbox_to_anchor=(0.3, 1))
        self.axes['ppg'].legend(loc='upper right', fontsize=8, framealpha=0.5)
        self.axes['motion'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['ref'].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values())
            
            # Get the minimum length across all EEG channels to ensure alignment
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values())
            
            # Calculate time axis based on actual timestamps
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                # Use actual time differences
                time_axis = timestamps - timestamps[-1]  # Make most recent = 0
                
                # Calculate display window based on time
                display_mask = time_axis >= -self.window_duration
                display_samples = np.sum(display_mask)
            else:
                return list(self.lines.values())
            
            # Update EEG with proper alignment and channel separation
            eeg_values_for_scaling = []
            channel_offsets = {'TP9': 0, 'AF7': 1, 'AF8': 2, 'TP10': 3}  # Vertical offsets
            
            for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:  # Fixed order
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        # Get aligned data
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
                        # Apply bandpass filter to remove noise and DC
                        if len(data_array) > 50:
                            # Simple high-pass filter to remove DC and low-frequency drift
                            # Using a moving average subtraction as a simple high-pass
                            window_size = min(50, len(data_array) // 4)
                            if window_size > 1:
                                # Calculate moving average
                                moving_avg = np.convolve(data_array, 
                                                       np.ones(window_size)/window_size, 
                                                       mode='same')
                                # Subtract to remove low frequencies
                                filtered_data = data_array - moving_avg
                            else:
                                filtered_data = data_array - np.mean(data_array)
                        else:
                            # For short data, just remove mean
                            filtered_data = data_array - np.mean(data_array)
                        
                        # Apply display mask
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        # Add channel offset for separation (optional - comment out for overlapped view)
                        # offset = channel_offsets[ch_name] * 50  # 50 μV separation
                        # display_data = display_data + offset
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            # Update PPG with same time alignment
            if 'PPG' in self.ppg_channels and len(self.ppg_channels['PPG']) > 0:
                ppg_data = np.array(list(self.ppg_channels['PPG']))
                if len(ppg_data) >= len(display_mask):
                    ppg_aligned = ppg_data[-len(display_mask):]
                    self.lines['ppg_PPG'].set_data(
                        time_axis[display_mask],
                        ppg_aligned[display_mask]
                    )
            
            # Update Motion with time alignment
            for ch_name, data in self.motion_channels.items():
                if len(data) > 0:
                    line_key = f'motion_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(data))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Reference with time alignment
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
            for ax in [self.axes['eeg'], self.axes['ppg'], 
                      self.axes['motion'], self.axes['ref']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            # Special handling for EEG to ensure proper scaling
            if eeg_values_for_scaling:
                # Use robust statistics to avoid outliers
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                
                # Set limits based on standard deviations from median
                y_range = 4 * eeg_std  # Show ±2 standard deviations
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
                
                # Add scale info
                self.axes['eeg'].text(0.02, 0.98, 
                                    f'Scale: ±{y_range/2:.1f} μV', 
                                    transform=self.axes['eeg'].transAxes, 
                                    fontsize=9,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Update info panel
            self._update_info_panel()
        
        return list(self.lines.values())

    
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
        
        # EEG stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'EEG (4ch):', fontsize=10, 
                             weight='bold', color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if ch in self.eeg_channels and len(self.eeg_channels[ch]) > 0:
                data = np.array(list(self.eeg_channels[ch])[-100:])
                # Show raw values (before DC removal)
                mean_val = np.mean(data)
                std_val = np.std(data)
                self.axes['info'].text(0.15, y_pos, f'{ch}:', fontsize=9,
                                     color=self.eeg_colors[ch],
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.4, y_pos, f'{mean_val:.1f}±{std_val:.1f}',
                                     fontsize=9, color='white',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.04
        
        # PPG stats
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, 'PPG:', fontsize=10,
                             weight='bold', color='#FF6B6B',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        if 'PPG' in self.ppg_channels and len(self.ppg_channels['PPG']) > 0:
            ppg_data = np.array(list(self.ppg_channels['PPG'])[-100:])
            mean_val = np.mean(ppg_data)
            self.axes['info'].text(0.15, y_pos, 'Value:', fontsize=9,
                                 color='white',
                                 transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.4, y_pos, f'{mean_val:.2e}',
                                 fontsize=9, color='#E74C3C',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
        
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
        
        self.axes['info'].text(0.15, y_pos, 'Buffer:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{len(self.timestamps)}/{self.buffer_size}',
                             fontsize=9, color='white',
                             transform=self.axes['info'].transAxes)
    
    def start(self):
        """Start the visualizer"""
        print("Starting Fixed Muse Visualizer...")
        print(f"Listening for OSC data on UDP port {self.port}")
        print("\nExpected data format:")
        print("  - EEG: 4 channels from 6 values (last 2 are NaN)")
        print("  - PPG: 1 valid channel from 3 values")
        print("  - Address format: 'username/datatype' (no leading /)")
        print("\nNOTE: EEG signals use high-pass filtering to remove DC offset")
        print("      Raw values (~600-800 μV) are shown in stats panel")
        print("      Displayed signals show variations around baseline")
        
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
                    for ch in self.ppg_channels.values():
                        ch.clear()
                    for ch in self.motion_channels.values():
                        ch.clear()
                    for ch in self.ref_channels.values():
                        ch.clear()
                    self.timestamps.clear()
                print("Buffers reset")
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=40,  # 25 FPS for smoother display
            blit=False,
            cache_frame_data=False
        )
        
        print("\nVisualization started!")
        print("First 5 EEG packets will be printed for verification.")
        print("\nKeyboard shortcuts:")
        print("  '+'/'-' : Increase/decrease time window")
        print("  'r'     : Reset buffers")
        print("  'q'     : Quit")
        
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


if __name__ == "__main__":
    # Create and start visualizer
    visualizer = MuseFixedVisualizer(
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