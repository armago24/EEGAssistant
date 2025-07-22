import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
from manualoscparser import EEGfNIRSParser
import time
from matplotlib.gridspec import GridSpec

class RealTimeVisualizer:
    def __init__(self, parser, update_interval=50, window_duration=10):
        self.parser = parser
        self.update_interval = update_interval  # milliseconds
        self.window_duration = window_duration  # seconds to display
        
        # Setup matplotlib with dark theme
        plt.style.use('dark_background')
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid layout
        gs = GridSpec(4, 2, figure=self.fig, height_ratios=[3, 3, 2, 1], width_ratios=[3, 1])
        
        # Main plots
        self.ax_eeg = self.fig.add_subplot(gs[0, 0])
        self.ax_fnirs = self.fig.add_subplot(gs[1, 0])
        self.ax_motion = self.fig.add_subplot(gs[2, 0])
        
        # Info panels
        self.ax_eeg_info = self.fig.add_subplot(gs[0, 1])
        self.ax_fnirs_info = self.fig.add_subplot(gs[1, 1])
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        
        # Status bar
        self.ax_status = self.fig.add_subplot(gs[3, :])
        
        self.setup_plots()
        
        # Data line objects
        self.eeg_lines = {}
        self.fnirs_lines = {}
        self.motion_lines = {}
        
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
        
        # Layout
        plt.tight_layout()
        
    def setup_plots(self):
        """Configure all plot aesthetics and labels"""
        # EEG plot setup
        self.ax_eeg.set_title('EEG Channels', fontsize=14, color='#4ECDC4', pad=10)
        self.ax_eeg.set_ylabel('Amplitude (μV)', fontsize=10)
        self.ax_eeg.set_xlabel('Time (s)', fontsize=10)
        self.ax_eeg.grid(True, alpha=0.2, linestyle='--')
        self.ax_eeg.set_facecolor('#1a1a1a')
        
        # fNIRS plot setup  
        self.ax_fnirs.set_title('fNIRS/Optical Channels', fontsize=14, color='#FF6B6B', pad=10)
        self.ax_fnirs.set_ylabel('Intensity (a.u.)', fontsize=10)
        self.ax_fnirs.set_xlabel('Time (s)', fontsize=10)
        self.ax_fnirs.grid(True, alpha=0.2, linestyle='--')
        self.ax_fnirs.set_facecolor('#1a1a1a')
        
        # Motion plot setup
        self.ax_motion.set_title('Motion Sensors', fontsize=12, color='#96CEB4', pad=10)
        self.ax_motion.set_ylabel('Acc (g) / Gyro (°/s)', fontsize=10)
        self.ax_motion.set_xlabel('Time (s)', fontsize=10)
        self.ax_motion.grid(True, alpha=0.2, linestyle='--')
        self.ax_motion.set_facecolor('#1a1a1a')
        
        # Info panels setup
        for ax in [self.ax_eeg_info, self.ax_fnirs_info, self.ax_stats]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#1a1a1a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        self.ax_eeg_info.set_title('EEG Stats', fontsize=12, color='#4ECDC4')
        self.ax_fnirs_info.set_title('fNIRS Stats', fontsize=12, color='#FF6B6B')
        self.ax_stats.set_title('System Stats', fontsize=12, color='#FFD93D')
        
        # Status bar setup
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        self.ax_status.set_facecolor('#0a0a0a')
        for spine in self.ax_status.spines.values():
            spine.set_visible(False)
        
    def initialize_plots(self):
        """Initialize plot lines based on detected channels"""
        # Wait for parser to detect data format
        wait_time = 0
        while not self.parser.data_format_detected and wait_time < 10:
            time.sleep(0.1)
            wait_time += 0.1
        
        if not self.parser.data_format_detected:
            print("Warning: Starting visualization without detected channels")
            return
        
        print("Initializing visualization...")
        
        # Setup EEG lines (excluding motion sensors)
        eeg_channels = [ch for ch in self.parser.eeg_channels.keys() 
                       if not ch.startswith('acc_') and not ch.startswith('gyro_')]
        
        for i, channel_name in enumerate(eeg_channels):
            color = self.eeg_colors.get(channel_name, '#FFFFFF')
            line, = self.ax_eeg.plot([], [], label=channel_name, color=color, 
                                    linewidth=1.5, alpha=0.9)
            self.eeg_lines[channel_name] = line
        
        # Setup fNIRS lines
        for i, channel_name in enumerate(sorted(self.parser.fnirs_channels.keys())):
            color = self.fnirs_colors[i % len(self.fnirs_colors)]
            line, = self.ax_fnirs.plot([], [], label=channel_name, color=color, 
                                      linewidth=1.5, alpha=0.9)
            self.fnirs_lines[channel_name] = line
        
        # Setup motion sensor lines
        motion_channels = [ch for ch in self.parser.eeg_channels.keys() 
                          if ch.startswith('acc_') or ch.startswith('gyro_')]
        
        for channel_name in motion_channels:
            color = self.motion_colors.get(channel_name, '#FFFFFF')
            line, = self.ax_motion.plot([], [], label=channel_name, color=color, 
                                       linewidth=1.5, alpha=0.9)
            self.motion_lines[channel_name] = line
        
        # Add legends
        if self.eeg_lines:
            self.ax_eeg.legend(loc='upper right', fontsize=8, framealpha=0.5)
        if self.fnirs_lines:
            # Only show legend if not too many channels
            if len(self.fnirs_lines) <= 8:
                self.ax_fnirs.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
        if self.motion_lines:
            self.ax_motion.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
            
        print(f"Initialized {len(self.eeg_lines)} EEG, {len(self.fnirs_lines)} fNIRS, "
              f"and {len(self.motion_lines)} motion channels")
    
    def update_plots(self, frame):
        """Update all plots with latest data"""
        try:
            eeg_data, fnirs_data, timestamps = self.parser.get_latest_data()
            
            if not timestamps:
                return list(self.eeg_lines.values()) + list(self.fnirs_lines.values()) + \
                       list(self.motion_lines.values())
            
            # Calculate time axis in seconds
            if len(timestamps) > 1:
                dt = np.diff(timestamps).mean()
                time_axis = np.arange(len(timestamps)) * dt
                time_axis = time_axis - time_axis[-1]  # Make most recent = 0
            else:
                time_axis = np.array([0])
            
            # Determine display window
            display_samples = int(self.window_duration / dt) if len(timestamps) > 1 else 100
            
            # Update EEG plots
            eeg_values_for_scaling = []
            for channel_name, line in self.eeg_lines.items():
                if channel_name in eeg_data and len(eeg_data[channel_name]) > 0:
                    data = np.array(eeg_data[channel_name])
                    
                    # Apply simple high-pass filter to remove DC offset
                    if len(data) > 10:
                        data = data - np.mean(data)
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    line.set_data(time_axis[-n_samples:], data[-n_samples:])
                    eeg_values_for_scaling.extend(data[-n_samples:])
            
            # Update fNIRS plots  
            fnirs_values_for_scaling = []
            for channel_name, line in self.fnirs_lines.items():
                if channel_name in fnirs_data and len(fnirs_data[channel_name]) > 0:
                    data = np.array(fnirs_data[channel_name])
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    line.set_data(time_axis[-n_samples:], data[-n_samples:])
                    fnirs_values_for_scaling.extend(data[-n_samples:])
            
            # Update motion plots
            motion_values_for_scaling = []
            for channel_name, line in self.motion_lines.items():
                if channel_name in eeg_data and len(eeg_data[channel_name]) > 0:
                    data = np.array(eeg_data[channel_name])
                    
                    # Update line data
                    n_samples = min(len(data), display_samples)
                    line.set_data(time_axis[-n_samples:], data[-n_samples:])
                    motion_values_for_scaling.extend(data[-n_samples:])
            
            # Update axis limits
            if len(time_axis) > 0:
                # Time axis
                for ax in [self.ax_eeg, self.ax_fnirs, self.ax_motion]:
                    ax.set_xlim(-self.window_duration, 0)
                
                # Y-axis scaling with margins
                if eeg_values_for_scaling:
                    eeg_min, eeg_max = np.percentile(eeg_values_for_scaling, [5, 95])
                    margin = (eeg_max - eeg_min) * 0.1
                    self.ax_eeg.set_ylim(eeg_min - margin, eeg_max + margin)
                
                if fnirs_values_for_scaling:
                    fnirs_min, fnirs_max = np.percentile(fnirs_values_for_scaling, [5, 95])
                    margin = (fnirs_max - fnirs_min) * 0.1
                    self.ax_fnirs.set_ylim(fnirs_min - margin, fnirs_max + margin)
                
                if motion_values_for_scaling:
                    motion_min, motion_max = np.percentile(motion_values_for_scaling, [5, 95])
                    margin = (motion_max - motion_min) * 0.1
                    self.ax_motion.set_ylim(motion_min - margin, motion_max + margin)
            
            # Update info panels
            self.update_info_panels(eeg_data, fnirs_data, timestamps)
            
            # Update sample rate
            current_time = time.time()
            self.sample_times.append(current_time)
            if len(self.sample_times) > 1:
                rate = len(self.sample_times) / (self.sample_times[-1] - self.sample_times[0])
                self.update_status_bar(rate, len(timestamps))
        
        except Exception as e:
            print(f"Plot update error: {e}")
        
        return list(self.eeg_lines.values()) + list(self.fnirs_lines.values()) + \
               list(self.motion_lines.values())
    
    def update_info_panels(self, eeg_data, fnirs_data, timestamps):
        """Update the information panels with statistics"""
        # Clear previous text
        self.ax_eeg_info.clear()
        self.ax_fnirs_info.clear()
        self.ax_stats.clear()
        
        # Reconfigure after clearing
        for ax in [self.ax_eeg_info, self.ax_fnirs_info, self.ax_stats]:
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
                      f'Port: {self.parser.port}')
        
        self.ax_status.text(0.5, 0.5, status_text, fontsize=10, 
                           ha='center', va='center', color='#888888',
                           transform=self.ax_status.transAxes)
    
    def start(self):
        """Start real-time visualization"""
        print("Starting real-time visualization...")
        
        # Start parser
        if not self.parser.start_receiving():
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
        self.parser.stop()

if __name__ == "__main__":
    # Create parser and visualizer
    parser = EEGfNIRSParser(port=8052, buffer_size=2000)
    parser.debug = False  # Turn off debug output for cleaner visualization
    
    visualizer = RealTimeVisualizer(parser, update_interval=50, window_duration=10)
    
    print("=" * 60)
    print("EEG/fNIRS Real-Time Visualizer")
    print("=" * 60)
    print(f"Waiting for UDP data on port {parser.port}...")
    print("Make sure your Muse is streaming to this port.")
    print("")
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if hasattr(visualizer, 'parser'):
            visualizer.parser.stop()