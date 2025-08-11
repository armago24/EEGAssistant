#!/usr/bin/env python3
"""
Muse S Athena Data Viewer
View and scroll through recorded EEG/fNIRS data with confusion markers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime, timedelta

class MuseDataViewer:
    def __init__(self):
        # Data storage
        self.data = None
        self.filename = None
        
        # View parameters
        self.window_duration = 10  # seconds
        self.current_time = 0
        
        # Plot elements
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.spectral_lines = {}
        self.confusion_lines = []
        self.time_slider = None
        
        # Color schemes (same as recorder)
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
        self.sample_rate = 256
        self.spectral_window_size = 512
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        self.max_freq = 70
        
    def load_data(self, filename=None):
        """Load NPZ data file"""
        if filename is None:
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                title="Select Muse Recording",
                initialdir=os.path.expanduser("~/Downloads"),
                filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
            )
            root.destroy()
            
            if not filename:
                return False
        
        try:
            print(f"\nLoading: {filename}")
            self.data = np.load(filename, allow_pickle=True)
            self.filename = filename
            
            # Print data info
            print(f"\n{'='*50}")
            print(f"DATA LOADED SUCCESSFULLY")
            print(f"File: {os.path.basename(filename)}")
            print(f"{'='*50}")
            
            # Check what data is available
            print("\nAvailable data:")
            for key in self.data.files:
                if key != 'metadata':
                    arr = self.data[key]
                    if hasattr(arr, 'shape'):
                        print(f"  {key}: {arr.shape}")
            
            # Get metadata
            if 'metadata' in self.data:
                metadata = self.data['metadata'].item()
                print(f"\nRecording info:")
                print(f"  Duration: {metadata.get('duration', 0):.1f} seconds")
                print(f"  Samples: {metadata.get('total_samples', 0)}")
                print(f"  Sample rate: {metadata.get('sample_rate', 256)} Hz")
                print(f"  Events: {metadata.get('total_events', 0)}")
                
                if 'start_time' in metadata:
                    start_dt = datetime.fromtimestamp(metadata['start_time'])
                    print(f"  Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check for confusion events
            if 'event_timestamps' in self.data and len(self.data['event_timestamps']) > 0:
                event_types = self.data['event_types']
                confusion_count = np.sum((event_types == 'confusion') | (event_types == 'c'))
                other_count = len(event_types) - confusion_count
                
                print(f"\nEvents in recording:")
                print(f"  🤔 Confusion markers: {confusion_count}")
                if other_count > 0:
                    print(f"  📍 Other markers: {other_count}")
                    
                # Show event timing
                if confusion_count > 0 and 'relative_timestamps' in self.data:
                    confusion_mask = (event_types == 'confusion') | (event_types == 'c')
                    confusion_times = self.data['event_timestamps'][confusion_mask]
                    start_time = self.data['timestamps'][0]
                    relative_times = confusion_times - start_time
                    
                    print(f"\nConfusion event times:")
                    for i, t in enumerate(relative_times[:10]):  # Show first 10
                        print(f"    {i+1}. At {t:.1f}s")
                    if len(relative_times) > 10:
                        print(f"    ... and {len(relative_times)-10} more")
            
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_plot(self):
        """Setup the visualization plot"""
        plt.style.use('dark_background')
        
        # Create figure with better layout for viewing
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid
        gs = GridSpec(7, 2, figure=self.fig,
                     height_ratios=[3, 3, 3, 2, 2, 1, 0.5],
                     width_ratios=[5, 1],
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
        
        # Set titles and labels
        self.axes['eeg'].set_title('EEG Channels', fontsize=14, color='#4ECDC4', pad=10)
        self.axes['eeg'].set_ylabel('Amplitude (μV)')
        
        self.axes['spectral'].set_title('Spectral Power', fontsize=14, color='#FFD93D', pad=10)
        self.axes['spectral'].set_ylabel('Power (dB)')
        
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
        self.axes['info'].set_title('Data Info', fontsize=14, color='#FFD93D', pad=10)
        
        # Add time slider
        ax_slider = plt.axes([0.15, 0.02, 0.65, 0.02], facecolor='#2a2a2a')
        total_duration = self.data['relative_timestamps'][-1] if 'relative_timestamps' in self.data else 100
        self.time_slider = Slider(
            ax_slider, 'Time', 0, total_duration,
            valinit=0, valstep=0.1,
            color='#4ECDC4'
        )
        self.time_slider.on_changed(self.update_view)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.02, 0.02, 0.05, 0.02])
        self.btn_prev = Button(ax_prev, '◀ Prev', color='#2a2a2a', hovercolor='#3a3a3a')
        self.btn_prev.on_clicked(self.prev_window)
        
        ax_next = plt.axes([0.08, 0.02, 0.05, 0.02])
        self.btn_next = Button(ax_next, 'Next ▶', color='#2a2a2a', hovercolor='#3a3a3a')
        self.btn_next.on_clicked(self.next_window)
        
        # Add window size buttons
        ax_zoom_in = plt.axes([0.82, 0.02, 0.04, 0.02])
        self.btn_zoom_in = Button(ax_zoom_in, 'Zoom In', color='#2a2a2a', hovercolor='#3a3a3a')
        self.btn_zoom_in.on_clicked(self.zoom_in)
        
        ax_zoom_out = plt.axes([0.87, 0.02, 0.04, 0.02])
        self.btn_zoom_out = Button(ax_zoom_out, 'Zoom Out', color='#2a2a2a', hovercolor='#3a3a3a')
        self.btn_zoom_out.on_clicked(self.zoom_out)
        
        # Add confusion navigation buttons
        ax_prev_conf = plt.axes([0.92, 0.02, 0.03, 0.02])
        self.btn_prev_conf = Button(ax_prev_conf, '◀🤔', color='#ff6644', hovercolor='#ff8866')
        self.btn_prev_conf.on_clicked(self.prev_confusion)
        
        ax_next_conf = plt.axes([0.96, 0.02, 0.03, 0.02])
        self.btn_next_conf = Button(ax_next_conf, '🤔▶', color='#ff6644', hovercolor='#ff8866')
        self.btn_next_conf.on_clicked(self.next_confusion)
        
        # Initialize lines
        self._initialize_lines()
        
        # Initial plot
        self.update_view(0)
        
    def _initialize_lines(self):
        """Initialize plot lines"""
        # EEG lines
        if 'eeg' in self.data and len(self.data['eeg']) > 0:
            for i, (ch, color) in enumerate(self.eeg_colors.items()):
                line, = self.axes['eeg'].plot([], [], label=ch, color=color,
                                             linewidth=1.5, alpha=0.95)
                self.lines[f'eeg_{ch}'] = line
            self.axes['eeg'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        
        # Spectral lines
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color,
                                              linewidth=1.5, alpha=0.9)
            self.spectral_lines[ch] = line
        self.axes['spectral'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        
        # fNIRS lines
        if 'fnirs' in self.data and len(self.data['fnirs']) > 0:
            for i, color in enumerate(self.fnirs_colors.values()):
                if i < 4:  # Only plot normalized values
                    line, = self.axes['fnirs'].plot([], [], label=f'Ch{i+1}',
                                                   color=color, linewidth=1.5, alpha=0.9)
                    self.lines[f'fnirs_Ch{i+1}'] = line
            self.axes['fnirs'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        
        # Motion lines
        if 'motion' in self.data and len(self.data['motion']) > 0:
            motion_colors = ['#3498DB', '#2ECC71', '#9B59B6']
            motion_labels = ['X', 'Y', 'Z']
            
            # Accelerometer
            for i, (label, color) in enumerate(zip(motion_labels, motion_colors)):
                line, = self.axes['motion'].plot([], [], label=label,
                                               color=color, linewidth=1.5, alpha=0.9)
                self.lines[f'acc_{label}'] = line
            self.axes['motion'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.7)
            
            # Gyroscope
            gyro_colors = ['#F39C12', '#E67E22', '#D35400']
            for i, (label, color) in enumerate(zip(motion_labels, gyro_colors)):
                line, = self.axes['gyro'].plot([], [], label=label,
                                              color=color, linewidth=1.5, alpha=0.9)
                self.lines[f'gyro_{label}'] = line
            self.axes['gyro'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.7)
        
        # Reference lines
        if 'ref' in self.data and len(self.data['ref']) > 0:
            ref_colors = ['#95A5A6', '#7F8C8D']
            ref_labels = ['DRL', 'REF']
            for i, (label, color) in enumerate(zip(ref_labels, ref_colors)):
                line, = self.axes['ref'].plot([], [], label=label,
                                             color=color, linewidth=1.5, alpha=0.9)
                self.lines[f'ref_{label}'] = line
            self.axes['ref'].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.7)
    
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
    
    def update_view(self, val):
        """Update all plots for the current time window"""
        self.current_time = self.time_slider.val
        
        # Get time window
        start_time = self.current_time
        end_time = start_time + self.window_duration
        
        # Get relative timestamps
        if 'relative_timestamps' not in self.data:
            return
        
        timestamps = self.data['relative_timestamps']
        
        # Find indices for time window
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        time_axis = timestamps[mask]
        
        if len(time_axis) == 0:
            return
        
        # Clear old confusion markers
        for line in self.confusion_lines:
            line.remove()
        self.confusion_lines = []
        
        # Add confusion markers if present
        if 'event_timestamps' in self.data and len(self.data['event_timestamps']) > 0:
            event_times = self.data['event_timestamps']
            event_types = self.data['event_types']
            start_timestamp = self.data['timestamps'][0]
            
            # Find confusion events in window
            confusion_mask = (event_types == 'confusion') | (event_types == 'c')
            confusion_times = event_times[confusion_mask] - start_timestamp
            
            # Draw vertical lines for confusion events
            for ax in [self.axes['eeg'], self.axes['fnirs'], self.axes['motion'], 
                      self.axes['gyro'], self.axes['ref']]:
                for conf_time in confusion_times:
                    if start_time <= conf_time <= end_time:
                        line = ax.axvline(x=conf_time, color='#ff6644', alpha=0.5,
                                        linewidth=2, linestyle='--', zorder=1)
                        self.confusion_lines.append(line)
        
        # Update EEG
        if 'eeg' in self.data and len(self.data['eeg']) > 0:
            eeg_data = self.data['eeg'][mask]
            if len(eeg_data) > 0:
                channels = ['TP9', 'AF7', 'AF8', 'TP10']
                for i, ch in enumerate(channels):
                    if f'eeg_{ch}' in self.lines:
                        # Apply high-pass filter
                        channel_data = eeg_data[:, i]
                        if len(channel_data) > 50:
                            window_size = min(50, len(channel_data) // 4)
                            if window_size > 1:
                                moving_avg = np.convolve(channel_data,
                                                       np.ones(window_size)/window_size,
                                                       mode='same')
                                filtered_data = channel_data - moving_avg
                            else:
                                filtered_data = channel_data - np.mean(channel_data)
                        else:
                            filtered_data = channel_data - np.mean(channel_data)
                        
                        self.lines[f'eeg_{ch}'].set_data(time_axis, filtered_data)
                
                # Update spectral analysis
                # Get a wider window for spectral analysis
                spectral_window = 4  # seconds
                spectral_start = max(0, self.current_time)
                spectral_end = min(timestamps[-1], self.current_time + spectral_window)
                spectral_mask = (timestamps >= spectral_start) & (timestamps <= spectral_end)
                spectral_data = self.data['eeg'][spectral_mask]
                
                if len(spectral_data) > self.spectral_window_size:
                    for i, ch in enumerate(channels):
                        if ch in self.spectral_lines:
                            channel_data = spectral_data[:, i]
                            # Remove DC
                            channel_data = channel_data - np.mean(channel_data)
                            frequencies, psd = self.compute_spectrum(channel_data, self.sample_rate)
                            if frequencies is not None:
                                self.spectral_lines[ch].set_data(frequencies, psd)
        
        # Update fNIRS
        if 'fnirs' in self.data and len(self.data['fnirs']) > 0:
            fnirs_data = self.data['fnirs'][mask]
            if len(fnirs_data) > 0:
                for i in range(4):  # Plot normalized values
                    if f'fnirs_Ch{i+1}' in self.lines:
                        self.lines[f'fnirs_Ch{i+1}'].set_data(time_axis, fnirs_data[:, i])
        
        # Update Motion
        if 'motion' in self.data and len(self.data['motion']) > 0:
            motion_data = self.data['motion'][mask]
            if len(motion_data) > 0:
                labels = ['X', 'Y', 'Z']
                # Accelerometer
                for i, label in enumerate(labels):
                    if f'acc_{label}' in self.lines:
                        self.lines[f'acc_{label}'].set_data(time_axis, motion_data[:, i])
                # Gyroscope
                for i, label in enumerate(labels):
                    if f'gyro_{label}' in self.lines:
                        self.lines[f'gyro_{label}'].set_data(time_axis, motion_data[:, i+3])
        
        # Update Reference
        if 'ref' in self.data and len(self.data['ref']) > 0:
            ref_data = self.data['ref'][mask]
            if len(ref_data) > 0:
                if 'ref_DRL' in self.lines:
                    self.lines['ref_DRL'].set_data(time_axis, ref_data[:, 0])
                if 'ref_REF' in self.lines:
                    self.lines['ref_REF'].set_data(time_axis, ref_data[:, 1])
        
        # Update axes limits
        for ax in [self.axes['eeg'], self.axes['fnirs'], self.axes['motion'],
                  self.axes['gyro'], self.axes['ref']]:
            ax.set_xlim(start_time, end_time)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        
        # Update spectral axes
        self.axes['spectral'].set_xlim(0, self.max_freq)
        self.axes['spectral'].relim()
        self.axes['spectral'].autoscale_view(scalex=False, scaley=True)
        
        # Update info panel
        self._update_info_panel()
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def _update_info_panel(self):
        """Update the info panel"""
        self.axes['info'].clear()
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        y_pos = 0.95
        
        # File info
        self.axes['info'].text(0.1, y_pos, 'File Info', fontsize=12, weight='bold',
                              color='#FFD93D', transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        if self.filename:
            self.axes['info'].text(0.1, y_pos, f'{os.path.basename(self.filename)[:25]}...',
                                  fontsize=9, color='white',
                                  transform=self.axes['info'].transAxes)
            y_pos -= 0.05
        
        # Current time window
        y_pos -= 0.03
        self.axes['info'].text(0.1, y_pos, 'Time Window', fontsize=11, weight='bold',
                              color='#4ECDC4', transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        self.axes['info'].text(0.1, y_pos, f'Start: {self.current_time:.1f}s',
                              fontsize=9, color='white',
                              transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.1, y_pos, f'End: {self.current_time + self.window_duration:.1f}s',
                              fontsize=9, color='white',
                              transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.1, y_pos, f'Window: {self.window_duration:.1f}s',
                              fontsize=9, color='white',
                              transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Stats for current window
        y_pos -= 0.03
        self.axes['info'].text(0.1, y_pos, 'Window Stats', fontsize=11, weight='bold',
                              color='#96CEB4', transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Count samples in window
        if 'relative_timestamps' in self.data:
            timestamps = self.data['relative_timestamps']
            mask = (timestamps >= self.current_time) & (timestamps <= self.current_time + self.window_duration)
            sample_count = np.sum(mask)
            
            self.axes['info'].text(0.1, y_pos, f'Samples: {sample_count}',
                                  fontsize=9, color='white',
                                  transform=self.axes['info'].transAxes)
            y_pos -= 0.04
        
        # Count confusion events in window
        if 'event_timestamps' in self.data and len(self.data['event_timestamps']) > 0:
            event_times = self.data['event_timestamps']
            event_types = self.data['event_types']
            start_timestamp = self.data['timestamps'][0]
            
            # Convert to relative times
            relative_event_times = event_times - start_timestamp
            
            # Find events in window
            window_mask = (relative_event_times >= self.current_time) & \
                         (relative_event_times <= self.current_time + self.window_duration)
            window_events = event_types[window_mask]
            
            confusion_count = np.sum((window_events == 'confusion') | (window_events == 'c'))
            other_count = len(window_events) - confusion_count
            
            if confusion_count > 0:
                self.axes['info'].text(0.1, y_pos, f'🤔 Confusion: {confusion_count}',
                                      fontsize=9, color='#ff6644',
                                      transform=self.axes['info'].transAxes)
                y_pos -= 0.04
            
            if other_count > 0:
                self.axes['info'].text(0.1, y_pos, f'📍 Other: {other_count}',
                                      fontsize=9, color='#66aaff',
                                      transform=self.axes['info'].transAxes)
                y_pos -= 0.04
        
        # Navigation hints
        y_pos -= 0.05
        self.axes['info'].text(0.1, y_pos, 'Navigation', fontsize=11, weight='bold',
                              color='#FFD93D', transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        hints = [
            ('←/→', 'Move 1s'),
            ('Shift+←/→', 'Move 10s'),
            ('↑/↓', 'Zoom'),
            ('c/C', 'Jump to confusion'),
            ('Home/End', 'Start/End'),
            ('Space', 'Play/Pause')
        ]
        
        for key, action in hints:
            self.axes['info'].text(0.1, y_pos, key, fontsize=8, color='#aaaaaa',
                                  transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.4, y_pos, action, fontsize=8, color='white',
                                  transform=self.axes['info'].transAxes)
            y_pos -= 0.035
        
    def prev_window(self, event=None):
        """Move to previous window"""
        new_time = max(0, self.current_time - self.window_duration * 0.5)
        self.time_slider.set_val(new_time)
    
    def next_window(self, event=None):
        """Move to next window"""
        max_time = self.data['relative_timestamps'][-1] - self.window_duration
        new_time = min(max_time, self.current_time + self.window_duration * 0.5)
        self.time_slider.set_val(new_time)
    
    def zoom_in(self, event=None):
        """Zoom in (decrease window size)"""
        self.window_duration = max(1, self.window_duration / 1.5)
        self.update_view(self.current_time)
        print(f"Window: {self.window_duration:.1f}s")
    
    def zoom_out(self, event=None):
        """Zoom out (increase window size)"""
        max_duration = self.data['relative_timestamps'][-1]
        self.window_duration = min(max_duration, self.window_duration * 1.5)
        self.update_view(self.current_time)
        print(f"Window: {self.window_duration:.1f}s")
    
    def prev_confusion(self, event=None):
        """Jump to previous confusion event"""
        if 'event_timestamps' not in self.data or len(self.data['event_timestamps']) == 0:
            return
        
        event_types = self.data['event_types']
        confusion_mask = (event_types == 'confusion') | (event_types == 'c')
        
        if not np.any(confusion_mask):
            return
        
        confusion_times = self.data['event_timestamps'][confusion_mask]
        start_timestamp = self.data['timestamps'][0]
        relative_confusion_times = confusion_times - start_timestamp
        
        # Find previous confusion event
        prev_times = relative_confusion_times[relative_confusion_times < self.current_time]
        if len(prev_times) > 0:
            target_time = prev_times[-1]
            # Center the confusion event in the window
            new_time = max(0, target_time - self.window_duration / 2)
            self.time_slider.set_val(new_time)
            print(f"Jumped to confusion at {target_time:.1f}s")
    
    def next_confusion(self, event=None):
        """Jump to next confusion event"""
        if 'event_timestamps' not in self.data or len(self.data['event_timestamps']) == 0:
            return
        
        event_types = self.data['event_types']
        confusion_mask = (event_types == 'confusion') | (event_types == 'c')
        
        if not np.any(confusion_mask):
            return
        
        confusion_times = self.data['event_timestamps'][confusion_mask]
        start_timestamp = self.data['timestamps'][0]
        relative_confusion_times = confusion_times - start_timestamp
        
        # Find next confusion event
        next_times = relative_confusion_times[relative_confusion_times > self.current_time + 0.1]
        if len(next_times) > 0:
            target_time = next_times[0]
            # Center the confusion event in the window
            max_time = self.data['relative_timestamps'][-1] - self.window_duration
            new_time = min(max_time, target_time - self.window_duration / 2)
            self.time_slider.set_val(new_time)
            print(f"Jumped to confusion at {target_time:.1f}s")
    
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'left':
            if event.key == 'shift+left':
                # Move 10 seconds back
                new_time = max(0, self.current_time - 10)
            else:
                # Move 1 second back
                new_time = max(0, self.current_time - 1)
            self.time_slider.set_val(new_time)
            
        elif event.key == 'right':
            max_time = self.data['relative_timestamps'][-1] - self.window_duration
            if event.key == 'shift+right':
                # Move 10 seconds forward
                new_time = min(max_time, self.current_time + 10)
            else:
                # Move 1 second forward
                new_time = min(max_time, self.current_time + 1)
            self.time_slider.set_val(new_time)
            
        elif event.key == 'up':
            self.zoom_in()
        elif event.key == 'down':
            self.zoom_out()
        elif event.key in ['c', 'C']:
            if event.key == 'C':
                self.prev_confusion()
            else:
                self.next_confusion()
        elif event.key == 'home':
            self.time_slider.set_val(0)
        elif event.key == 'end':
            max_time = self.data['relative_timestamps'][-1] - self.window_duration
            self.time_slider.set_val(max_time)
        elif event.key == ' ':
            # Space bar - toggle play/pause (simple auto-scroll)
            if not hasattr(self, 'playing'):
                self.playing = False
            self.playing = not self.playing
            if self.playing:
                self.play()
    
    def play(self):
        """Simple playback function"""
        import time as time_module
        while self.playing:
            max_time = self.data['relative_timestamps'][-1] - self.window_duration
            if self.current_time >= max_time:
                self.playing = False
                break
            new_time = min(max_time, self.current_time + 0.5)
            self.time_slider.set_val(new_time)
            plt.pause(0.05)
    
    def run(self):
        """Main entry point"""
        print("\n" + "="*60)
        print("   MUSE S ATHENA DATA VIEWER")
        print("="*60)
        print("\n📂 Select a .npz recording file to view")
        
        if not self.load_data():
            print("No file selected. Exiting.")
            return
        
        print("\n🎮 CONTROLS:")
        print("  • Drag slider to navigate through time")
        print("  • Use ◀/▶ buttons to move by half window")
        print("  • Use 🤔 buttons to jump between confusion events")
        print("  • Zoom In/Out to change time window")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("  ←/→        : Move 1 second")
        print("  Shift+←/→  : Move 10 seconds")
        print("  ↑/↓        : Zoom in/out")
        print("  c          : Next confusion event")
        print("  C          : Previous confusion event")
        print("  Home/End   : Jump to start/end")
        print("  Space      : Play/Pause auto-scroll")
        
        print("\n🤔 Confusion events are marked with orange dashed lines")
        print("="*60 + "\n")
        
        self.setup_plot()
        
        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.show()


if __name__ == "__main__":
    viewer = MuseDataViewer()
    viewer.run()