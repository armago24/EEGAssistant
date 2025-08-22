#!/usr/bin/env python3
"""
EEG Confusion Event Analyzer
Analyzes EEG data to find signatures associated with word and sentence confusion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ConfusionAnalyzer:
    def __init__(self, filepath):
        """Load and prepare data for analysis"""
        print(f"\n{'='*60}")
        print("EEG CONFUSION EVENT ANALYZER")
        print(f"{'='*60}\n")
        
        # Load data
        print(f"Loading: {filepath}")
        self.data = np.load(filepath, allow_pickle=True)
        
        # Extract main data arrays
        self.timestamps = self.data['timestamps']
        self.relative_timestamps = self.data['relative_timestamps']
        self.eeg = self.data['eeg']
        self.event_timestamps = self.data['event_timestamps']
        self.event_types = self.data['event_types']
        
        # Get metadata
        self.metadata = self.data['metadata'].item() if 'metadata' in self.data else {}
        self.sample_rate = self.metadata.get('sample_rate', 256)
        self.channels = self.metadata.get('eeg_channels', ['TP9', 'AF7', 'AF8', 'TP10'])
        
        # Print data summary
        print(f"\nData Summary:")
        print(f"  Duration: {self.relative_timestamps[-1]:.1f} seconds")
        print(f"  Samples: {len(self.timestamps)}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  EEG shape: {self.eeg.shape}")
        print(f"  Channels: {', '.join(self.channels)}")
        
        # Count events
        self.word_events = []
        self.sentence_events = []
        
        for i, (timestamp, event_type) in enumerate(zip(self.event_timestamps, self.event_types)):
            relative_time = timestamp - self.timestamps[0]
            if event_type.lower() == 'c':
                self.word_events.append((timestamp, relative_time))
            elif event_type.lower() == 's':
                self.sentence_events.append((timestamp, relative_time))
        
        print(f"\nEvents:")
        print(f"  Word confusion (C): {len(self.word_events)} events")
        print(f"  Sentence confusion (S): {len(self.sentence_events)} events")
        
        # Preprocess EEG
        self.preprocess_eeg()
        
    def preprocess_eeg(self):
        """Basic preprocessing: remove DC offset and apply bandpass filter"""
        print("\nPreprocessing EEG data...")
        
        # Remove DC offset
        self.eeg_processed = self.eeg - np.mean(self.eeg, axis=0)
        
        # Apply bandpass filter (0.5-50 Hz to remove drift and high-freq noise)
        nyquist = self.sample_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        if low < 1 and high < 1:
            b, a = signal.butter(4, [low, high], btype='band')
            
            for ch in range(self.eeg.shape[1]):
                if not np.all(np.isnan(self.eeg[:, ch])):
                    self.eeg_processed[:, ch] = signal.filtfilt(b, a, self.eeg[:, ch])
        
        print("  ✓ DC offset removed")
        print("  ✓ Bandpass filtered (0.5-50 Hz)")
    
    def extract_epochs(self, event_times, pre_sec=0.5, post_sec=1.5):
        """Extract EEG epochs around events"""
        pre_samples = int(pre_sec * self.sample_rate)
        post_samples = int(post_sec * self.sample_rate)
        
        epochs = []
        valid_events = []
        
        for event_time, rel_time in event_times:
            # Find closest sample
            idx = np.argmin(np.abs(self.timestamps - event_time))
            
            # Check bounds
            if idx - pre_samples >= 0 and idx + post_samples < len(self.timestamps):
                epoch = self.eeg_processed[idx - pre_samples:idx + post_samples, :]
                epochs.append(epoch)
                valid_events.append(rel_time)
        
        if epochs:
            return np.array(epochs), valid_events
        else:
            return None, []
    
    def extract_baseline(self, event_times, duration_sec=2.0, offset_sec=3.0):
        """Extract baseline periods (away from any events)"""
        samples_per_segment = int(duration_sec * self.sample_rate)
        offset_samples = int(offset_sec * self.sample_rate)
        
        baseline_segments = []
        
        # Find periods at least offset_sec away from any event
        all_event_times = [t for t, _ in event_times]
        
        for i in range(0, len(self.timestamps) - samples_per_segment, samples_per_segment // 2):
            segment_start_time = self.timestamps[i]
            segment_end_time = self.timestamps[i + samples_per_segment - 1]
            
            # Check if this segment is far enough from all events
            is_baseline = True
            for event_time in all_event_times:
                if abs(segment_start_time - event_time) < offset_sec or \
                   abs(segment_end_time - event_time) < offset_sec:
                    is_baseline = False
                    break
            
            if is_baseline:
                segment = self.eeg_processed[i:i + samples_per_segment, :]
                baseline_segments.append(segment)
        
        return baseline_segments
    
    def compute_erp(self):
        """Compute Event-Related Potentials"""
        print("\n" + "="*40)
        print("COMPUTING EVENT-RELATED POTENTIALS")
        print("="*40)
        
        # Extract epochs for word confusion
        word_epochs, word_times = self.extract_epochs(self.word_events)
        if word_epochs is not None:
            self.word_erp = np.mean(word_epochs, axis=0)
            self.word_erp_std = np.std(word_epochs, axis=0)
            print(f"Word confusion: {len(word_epochs)} epochs extracted")
        else:
            self.word_erp = None
            print("Word confusion: No valid epochs")
        
        # Extract epochs for sentence confusion
        sentence_epochs, sentence_times = self.extract_epochs(self.sentence_events)
        if sentence_epochs is not None:
            self.sentence_erp = np.mean(sentence_epochs, axis=0)
            self.sentence_erp_std = np.std(sentence_epochs, axis=0)
            print(f"Sentence confusion: {len(sentence_epochs)} epochs extracted")
        else:
            self.sentence_erp = None
            print("Sentence confusion: No valid epochs")
        
        # Extract baseline
        all_events = self.word_events + self.sentence_events
        baseline_segments = self.extract_baseline(all_events)
        if baseline_segments:
            # Trim baseline segments to match epoch length
            epoch_len = len(self.word_erp) if self.word_erp is not None else len(self.sentence_erp)
            baseline_segments = [seg[:epoch_len, :] for seg in baseline_segments if len(seg) >= epoch_len]
            if baseline_segments:
                self.baseline_erp = np.mean(baseline_segments, axis=0)
                self.baseline_erp_std = np.std(baseline_segments, axis=0)
                print(f"Baseline: {len(baseline_segments)} segments extracted")
            else:
                self.baseline_erp = None
        else:
            self.baseline_erp = None
            print("Baseline: No valid segments")
        
        # Store epochs for later analysis
        self.word_epochs = word_epochs
        self.sentence_epochs = sentence_epochs
    
    def compute_spectral_features(self):
        """Compute frequency domain features"""
        print("\n" + "="*40)
        print("COMPUTING SPECTRAL FEATURES")
        print("="*40)
        
        freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        def compute_band_power(epochs):
            """Compute average power in each frequency band"""
            if epochs is None:
                return None
            
            band_powers = {band: [] for band in freq_bands}
            
            for epoch in epochs:
                for ch in range(epoch.shape[1]):
                    # Compute PSD using Welch's method
                    freqs, psd = signal.welch(epoch[:, ch], fs=self.sample_rate, nperseg=min(len(epoch), 256))
                    
                    for band, (low, high) in freq_bands.items():
                        band_mask = (freqs >= low) & (freqs <= high)
                        band_power = np.mean(psd[band_mask])
                        band_powers[band].append(band_power)
            
            # Average across all epochs and channels
            for band in band_powers:
                band_powers[band] = np.mean(band_powers[band])
            
            return band_powers
        
        # Compute for each condition
        self.word_band_powers = compute_band_power(self.word_epochs)
        self.sentence_band_powers = compute_band_power(self.sentence_epochs)
        
        # Compute baseline band power
        if hasattr(self, 'baseline_erp') and self.baseline_erp is not None:
            baseline_epochs = self.extract_baseline(self.word_events + self.sentence_events)
            if baseline_epochs:
                self.baseline_band_powers = compute_band_power(np.array(baseline_epochs))
            else:
                self.baseline_band_powers = None
        else:
            self.baseline_band_powers = None
        
        # Print results
        if self.word_band_powers:
            print("\nWord Confusion - Band Powers:")
            for band, power in self.word_band_powers.items():
                baseline_power = self.baseline_band_powers[band] if self.baseline_band_powers else 0
                change = ((power - baseline_power) / baseline_power * 100) if baseline_power else 0
                print(f"  {band:6s}: {power:.2e} {f'({change:+.1f}% vs baseline)' if baseline_power else ''}")
        
        if self.sentence_band_powers:
            print("\nSentence Confusion - Band Powers:")
            for band, power in self.sentence_band_powers.items():
                baseline_power = self.baseline_band_powers[band] if self.baseline_band_powers else 0
                change = ((power - baseline_power) / baseline_power * 100) if baseline_power else 0
                print(f"  {band:6s}: {power:.2e} {f'({change:+.1f}% vs baseline)' if baseline_power else ''}")
    
    def plot_results(self):
        """Generate comprehensive visualization"""
        print("\n" + "="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40)
        
        # Setup figure
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('EEG Confusion Analysis Results', fontsize=16, fontweight='bold')
        
        # Time axis for ERPs
        if self.word_erp is not None or self.sentence_erp is not None:
            erp_len = len(self.word_erp) if self.word_erp is not None else len(self.sentence_erp)
            time_axis = np.linspace(-0.5, 1.5, erp_len)
        
        # 1. Raw EEG with event markers
        ax1 = plt.subplot(4, 2, (1, 2))
        
        # Plot first 60 seconds of data
        plot_duration = min(60, self.relative_timestamps[-1])
        plot_samples = int(plot_duration * self.sample_rate)
        
        for ch_idx, ch_name in enumerate(self.channels):
            offset = ch_idx * 50  # Offset for visualization
            ax1.plot(self.relative_timestamps[:plot_samples], 
                    self.eeg_processed[:plot_samples, ch_idx] + offset,
                    label=ch_name, alpha=0.7, linewidth=0.5)
        
        # Mark events
        for _, rel_time in self.word_events:
            if rel_time <= plot_duration:
                ax1.axvline(x=rel_time, color='orange', alpha=0.5, linestyle='--', label='Word' if _ == self.word_events[0][0] else '')
        
        for _, rel_time in self.sentence_events:
            if rel_time <= plot_duration:
                ax1.axvline(x=rel_time, color='red', alpha=0.5, linestyle='--', label='Sentence' if _ == self.sentence_events[0][0] else '')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (µV)')
        ax1.set_title('Raw EEG Data with Confusion Events (first 60s)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Event-Related Potentials - Word Confusion
        if self.word_erp is not None:
            ax2 = plt.subplot(4, 2, 3)
            for ch_idx, ch_name in enumerate(self.channels):
                ax2.plot(time_axis, self.word_erp[:, ch_idx], label=ch_name, linewidth=2)
                ax2.fill_between(time_axis, 
                                self.word_erp[:, ch_idx] - self.word_erp_std[:, ch_idx],
                                self.word_erp[:, ch_idx] + self.word_erp_std[:, ch_idx],
                                alpha=0.2)
            
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Time relative to event (s)')
            ax2.set_ylabel('Amplitude (µV)')
            ax2.set_title(f'Word Confusion ERP (n={len(self.word_events)})')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # 3. Event-Related Potentials - Sentence Confusion
        if self.sentence_erp is not None:
            ax3 = plt.subplot(4, 2, 4)
            for ch_idx, ch_name in enumerate(self.channels):
                ax3.plot(time_axis, self.sentence_erp[:, ch_idx], label=ch_name, linewidth=2)
                ax3.fill_between(time_axis,
                                self.sentence_erp[:, ch_idx] - self.sentence_erp_std[:, ch_idx],
                                self.sentence_erp[:, ch_idx] + self.sentence_erp_std[:, ch_idx],
                                alpha=0.2)
            
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xlabel('Time relative to event (s)')
            ax3.set_ylabel('Amplitude (µV)')
            ax3.set_title(f'Sentence Confusion ERP (n={len(self.sentence_events)})')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # 4. Difference from baseline
        if self.baseline_erp is not None:
            ax4 = plt.subplot(4, 2, 5)
            
            if self.word_erp is not None:
                word_diff = self.word_erp - self.baseline_erp[:len(self.word_erp)]
                for ch_idx, ch_name in enumerate(self.channels):
                    ax4.plot(time_axis, word_diff[:, ch_idx], 
                           label=f'{ch_name} (word)', linewidth=1.5, linestyle='-')
            
            if self.sentence_erp is not None:
                sentence_diff = self.sentence_erp - self.baseline_erp[:len(self.sentence_erp)]
                for ch_idx, ch_name in enumerate(self.channels):
                    ax4.plot(time_axis, sentence_diff[:, ch_idx], 
                           label=f'{ch_name} (sent)', linewidth=1.5, linestyle='--')
            
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Time relative to event (s)')
            ax4.set_ylabel('Difference from baseline (µV)')
            ax4.set_title('Confusion vs Baseline Difference')
            ax4.legend(loc='upper right', fontsize=6, ncol=2)
            ax4.grid(True, alpha=0.3)
        
        # 5. Spectral Power Comparison
        if self.word_band_powers or self.sentence_band_powers:
            ax5 = plt.subplot(4, 2, 6)
            
            bands = list(self.word_band_powers.keys()) if self.word_band_powers else list(self.sentence_band_powers.keys())
            x_pos = np.arange(len(bands))
            width = 0.25
            
            if self.baseline_band_powers:
                baseline_values = [self.baseline_band_powers[band] for band in bands]
                ax5.bar(x_pos - width, baseline_values, width, label='Baseline', color='gray', alpha=0.7)
            
            if self.word_band_powers:
                word_values = [self.word_band_powers[band] for band in bands]
                ax5.bar(x_pos, word_values, width, label='Word Confusion', color='orange', alpha=0.7)
            
            if self.sentence_band_powers:
                sentence_values = [self.sentence_band_powers[band] for band in bands]
                ax5.bar(x_pos + width, sentence_values, width, label='Sentence Confusion', color='red', alpha=0.7)
            
            ax5.set_xlabel('Frequency Band')
            ax5.set_ylabel('Power')
            ax5.set_title('Spectral Power by Frequency Band')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(bands)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_yscale('log')
        
        # 6. Time-frequency spectrogram for events
        if self.word_epochs is not None and len(self.word_epochs) > 0:
            ax6 = plt.subplot(4, 2, 7)
            
            # Average across channels and epochs
            avg_signal = np.mean(self.word_epochs, axis=(0, 2))
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(avg_signal, fs=self.sample_rate, nperseg=64, noverlap=56)
            
            # Plot
            t_shifted = t - 0.5  # Shift to align with event at t=0
            pcm = ax6.pcolormesh(t_shifted, f[f <= 50], 10 * np.log10(Sxx[f <= 50, :]), 
                                shading='gouraud', cmap='viridis')
            ax6.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            ax6.set_ylabel('Frequency (Hz)')
            ax6.set_xlabel('Time relative to event (s)')
            ax6.set_title('Word Confusion - Time-Frequency Analysis')
            plt.colorbar(pcm, ax=ax6, label='Power (dB)')
        
        # 7. Statistical significance
        if self.word_epochs is not None and self.sentence_epochs is not None:
            ax7 = plt.subplot(4, 2, 8)
            
            # Compute peak amplitudes for each epoch (0-500ms post-event)
            peak_window = slice(int(0.5 * self.sample_rate), int(1.0 * self.sample_rate))
            
            word_peaks = np.max(np.abs(self.word_epochs[:, peak_window, :]), axis=1)
            sentence_peaks = np.max(np.abs(self.sentence_epochs[:, peak_window, :]), axis=1)
            
            # Plot distributions
            for ch_idx, ch_name in enumerate(self.channels):
                ax7.hist(word_peaks[:, ch_idx], bins=15, alpha=0.5, label=f'{ch_name} (word)', 
                        density=True, color=f'C{ch_idx}')
                ax7.hist(sentence_peaks[:, ch_idx], bins=15, alpha=0.5, label=f'{ch_name} (sent)', 
                        density=True, color=f'C{ch_idx}', linestyle='--', histtype='step', linewidth=2)
            
            ax7.set_xlabel('Peak Amplitude (µV)')
            ax7.set_ylabel('Density')
            ax7.set_title('Peak Amplitude Distributions (0-500ms)')
            ax7.legend(loc='upper right', fontsize=6, ncol=2)
            ax7.grid(True, alpha=0.3)
            
            # Perform t-tests
            print("\nStatistical Tests (Word vs Sentence):")
            for ch_idx, ch_name in enumerate(self.channels):
                t_stat, p_val = stats.ttest_ind(word_peaks[:, ch_idx], sentence_peaks[:, ch_idx])
                print(f"  {ch_name}: t={t_stat:.2f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Visualizations complete")
    
    def find_patterns(self):
        """Look for distinctive patterns in confusion events"""
        print("\n" + "="*40)
        print("PATTERN DETECTION")
        print("="*40)
        
        if self.word_epochs is None and self.sentence_epochs is None:
            print("No epochs available for pattern detection")
            return
        
        # 1. Check for N400-like response (negative deflection ~400ms)
        n400_window = slice(int(0.3 * self.sample_rate), int(0.5 * self.sample_rate))
        
        if self.word_erp is not None:
            n400_word = np.mean(self.word_erp[n400_window, :], axis=0)
            print(f"\nWord Confusion - N400 window (300-500ms):")
            for ch_idx, ch_name in enumerate(self.channels):
                print(f"  {ch_name}: {n400_word[ch_idx]:.2f} µV")
        
        if self.sentence_erp is not None:
            n400_sentence = np.mean(self.sentence_erp[n400_window, :], axis=0)
            print(f"\nSentence Confusion - N400 window (300-500ms):")
            for ch_idx, ch_name in enumerate(self.channels):
                print(f"  {ch_name}: {n400_sentence[ch_idx]:.2f} µV")
        
        # 2. Check for P600-like response (positive deflection ~600ms)
        p600_window = slice(int(0.5 * self.sample_rate), int(0.8 * self.sample_rate))
        
        if self.sentence_erp is not None:
            p600_sentence = np.mean(self.sentence_erp[p600_window, :], axis=0)
            print(f"\nSentence Confusion - P600 window (500-800ms):")
            for ch_idx, ch_name in enumerate(self.channels):
                print(f"  {ch_name}: {p600_sentence[ch_idx]:.2f} µV")
        
        # 3. Pre-event activity (anticipation?)
        pre_window = slice(int(0.2 * self.sample_rate), int(0.45 * self.sample_rate))
        
        if self.word_erp is not None:
            pre_word = np.mean(np.abs(self.word_erp[pre_window, :]), axis=0)
            print(f"\nPre-event activity (-300 to -50ms):")
            print(f"  Word: {np.mean(pre_word):.2f} µV (avg)")
        
        if self.sentence_erp is not None:
            pre_sentence = np.mean(np.abs(self.sentence_erp[pre_window, :]), axis=0)
            print(f"  Sentence: {np.mean(pre_sentence):.2f} µV (avg)")
        
        # 4. Frontal vs Temporal differences
        if self.word_erp is not None or self.sentence_erp is not None:
            print(f"\nSpatial Patterns:")
            
            frontal_channels = ['AF7', 'AF8']
            temporal_channels = ['TP9', 'TP10']
            
            frontal_idx = [self.channels.index(ch) for ch in frontal_channels if ch in self.channels]
            temporal_idx = [self.channels.index(ch) for ch in temporal_channels if ch in self.channels]
            
            if self.word_erp is not None and frontal_idx and temporal_idx:
                frontal_power = np.mean(np.abs(self.word_erp[:, frontal_idx]))
                temporal_power = np.mean(np.abs(self.word_erp[:, temporal_idx]))
                print(f"  Word - Frontal/Temporal ratio: {frontal_power/temporal_power:.2f}")
            
            if self.sentence_erp is not None and frontal_idx and temporal_idx:
                frontal_power = np.mean(np.abs(self.sentence_erp[:, frontal_idx]))
                temporal_power = np.mean(np.abs(self.sentence_erp[:, temporal_idx]))
                print(f"  Sentence - Frontal/Temporal ratio: {frontal_power/temporal_power:.2f}")

def main():
    """Main analysis function"""
    import sys
    import glob
    
    # Find NPZ files
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for NPZ files in current directory and Downloads
        search_paths = [
            "*.npz",
            "~/Downloads/*.npz",
            os.path.expanduser("~/Downloads/*.npz")
        ]
        
        npz_files = []
        for path in search_paths:
            npz_files.extend(glob.glob(path))
        
        if not npz_files:
            print("No NPZ files found. Please specify a file path.")
            return
        
        # Show available files
        print("Available NPZ files:")
        for i, f in enumerate(npz_files):
            size = os.path.getsize(f) / 1024
            print(f"  {i+1}. {os.path.basename(f)} ({size:.1f} KB)")
        
        # Ask user to select
        if len(npz_files) == 1:
            filepath = npz_files[0]
            print(f"\nUsing: {filepath}")
        else:
            try:
                choice = int(input("\nSelect file number: ")) - 1
                filepath = npz_files[choice]
            except (ValueError, IndexError):
                print("Invalid selection")
                return
    
    # Run analysis
    analyzer = ConfusionAnalyzer(filepath)
    analyzer.compute_erp()
    analyzer.compute_spectral_features()
    analyzer.find_patterns()
    analyzer.plot_results()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings to look for:")
    print("  • N400: Negative deflection ~400ms (semantic processing)")
    print("  • P600: Positive deflection ~600ms (syntactic reanalysis)")
    print("  • Theta power increase: Associated with cognitive effort")
    print("  • Frontal activity: Executive control during confusion")
    print("\nNote: Consumer EEG may show subtle effects.")
    print("More training data will improve pattern detection!")

if __name__ == "__main__":
    main()