#!/usr/bin/env python3
"""
Word-Level EEG/fNIRS Confusion Detector - Modified Version
Excludes pre-click data to test if model is detecting click intent vs confusion
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class WordConfusionDetector:
    def __init__(self, filepath, exclude_preclick_time=0.0):
        """Initialize and load data
        
        Args:
            filepath: Path to NPZ file
            exclude_preclick_time: Time in seconds to exclude before click (default 0.0)
        """
        self.exclude_preclick_time = exclude_preclick_time
        
        print(f"\n{'='*60}")
        print("WORD-LEVEL CONFUSION DETECTOR")
        if exclude_preclick_time > 0:
            print(f"EXCLUDING {exclude_preclick_time}s PRE-CLICK DATA")
        print(f"{'='*60}\n")
        
        # Load data
        print(f"Loading: {filepath}")
        self.data = np.load(filepath, allow_pickle=True)
        
        # Extract arrays
        self.timestamps = self.data['timestamps']
        self.eeg = self.data['eeg']
        self.fnirs = self.data['fnirs'] if 'fnirs' in self.data else None
        self.motion = self.data['motion'] if 'motion' in self.data else None
        
        # Event data - now includes clicked words
        self.event_timestamps = self.data['event_timestamps']
        self.event_types = self.data['event_types']
        self.event_words = self.data['event_words'] if 'event_words' in self.data else None
        
        # Continuous word tracking
        self.tracked_words = self.data['words'] if 'words' in self.data else None
        
        # Metadata
        self.metadata = self.data['metadata'].item() if 'metadata' in self.data else {}
        self.sample_rate = self.metadata.get('sample_rate', 256)
        self.channels = self.metadata.get('eeg_channels', ['TP9', 'AF7', 'AF8', 'TP10'])
        self.text_passage = self.metadata.get('text_passage', '')
        
        print(f"  Duration: {(self.timestamps[-1] - self.timestamps[0]):.1f}s")
        print(f"  Samples: {len(self.timestamps)}")
        print(f"  EEG Channels: {', '.join(self.channels)}")
        if self.fnirs is not None:
            print(f"  fNIRS Channels: {self.fnirs.shape[1]} (4 normalized + 4 raw)")
        
        # Process events and words
        self.process_events()
        
        # Preprocess signals
        self.preprocess_signals()
        
    def process_events(self):
        """Process clicked words and build confusion word database"""
        self.confused_words = defaultdict(list)  # word -> list of (timestamp, type)
        self.all_words_timeline = []  # (timestamp, word) for all tracked words
        
        # Process clicked confusion events
        if self.event_words is not None:
            print(f"\nProcessing clicked words...")
            word_confusion_count = 0
            sentence_confusion_count = 0
            
            for timestamp, event_type, word in zip(self.event_timestamps, self.event_types, self.event_words):
                if word and isinstance(word, str) and word.strip():
                    clean_word = word.strip().lower()
                    
                    if event_type == 'word_confusion':
                        self.confused_words[clean_word].append((timestamp, 'word'))
                        word_confusion_count += 1
                    elif event_type == 'sentence_confusion':
                        self.confused_words[clean_word].append((timestamp, 'sentence'))
                        sentence_confusion_count += 1
            
            print(f"  Word confusion clicks: {word_confusion_count}")
            print(f"  Sentence confusion clicks: {sentence_confusion_count}")
            print(f"  Unique confused words: {len(self.confused_words)}")
            
            if self.confused_words:
                print(f"\nTop confused words:")
                for word, events in sorted(self.confused_words.items(), 
                                          key=lambda x: len(x[1]), reverse=True)[:10]:
                    print(f"  '{word}': {len(events)} times")
        
        # Build timeline of all words being read
        if self.tracked_words is not None:
            print(f"\nProcessing word timeline...")
            unique_tracked = set()
            
            for timestamp, word in zip(self.timestamps, self.tracked_words):
                if word and isinstance(word, str) and word.strip():
                    clean_word = word.strip().lower()
                    self.all_words_timeline.append((timestamp, clean_word))
                    unique_tracked.add(clean_word)
            
            print(f"  Total word observations: {len(self.all_words_timeline)}")
            print(f"  Unique words tracked: {len(unique_tracked)}")
    
    def preprocess_signals(self):
        """Preprocess EEG and fNIRS data"""
        print("\nPreprocessing signals...")
        
        # EEG preprocessing
        self.eeg_processed = self.eeg - np.mean(self.eeg, axis=0)
        
        # Bandpass filter (0.5-50 Hz)
        nyquist = self.sample_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        if low < 1 and high < 1:
            b, a = signal.butter(4, [low, high], btype='band')
            for ch in range(self.eeg.shape[1]):
                if not np.all(np.isnan(self.eeg[:, ch])):
                    self.eeg_processed[:, ch] = signal.filtfilt(b, a, self.eeg[:, ch])
        
        # Notch filters
        for freq in [60, 120]:
            if freq < nyquist:
                notch_freq = freq / nyquist
                b_notch, a_notch = signal.iirnotch(notch_freq, Q=30)
                for ch in range(self.eeg.shape[1]):
                    if not np.all(np.isnan(self.eeg_processed[:, ch])):
                        self.eeg_processed[:, ch] = signal.filtfilt(b_notch, a_notch, self.eeg_processed[:, ch])
        
        # fNIRS preprocessing
        if self.fnirs is not None:
            print("  Processing fNIRS data...")
            self.fnirs_processed = np.copy(self.fnirs)
            
            # Separate normalized and raw channels
            self.fnirs_norm = self.fnirs[:, :4]  # First 4 are normalized
            self.fnirs_raw = self.fnirs[:, 4:]   # Last 4 are raw
            
            # Apply lowpass filter to fNIRS (hemodynamic response is slow)
            # Cutoff at 0.5 Hz to capture hemodynamic changes
            if 0.5 < nyquist:
                fnirs_low = 0.5 / nyquist
                b_low, a_low = signal.butter(4, fnirs_low, btype='low')
                
                for ch in range(self.fnirs_norm.shape[1]):
                    if not np.all(np.isnan(self.fnirs_norm[:, ch])):
                        # Detrend first
                        self.fnirs_norm[:, ch] = signal.detrend(self.fnirs_norm[:, ch])
                        # Then filter
                        self.fnirs_norm[:, ch] = signal.filtfilt(b_low, a_low, self.fnirs_norm[:, ch])
        
        print("  ✓ Preprocessing complete")
    
    def extract_word_features(self, word, timestamp, window_size=2.0, is_confusion_event=False):
        """Extract features for a specific word occurrence
        
        Args:
            word: The word to analyze
            timestamp: When the word was encountered
            window_size: Total window size in seconds
            is_confusion_event: If True, this is a confusion click event
        """
        features = []
        
        # Adjust window based on whether this is a confusion event
        if is_confusion_event and self.exclude_preclick_time > 0:
            # For confusion events, exclude pre-click data
            # Window: [timestamp - window_size/2, timestamp - exclude_preclick_time]
            start_time = timestamp - window_size/2
            end_time = timestamp - self.exclude_preclick_time
            
            # Ensure we have a reasonable window
            if end_time <= start_time:
                # If exclusion is too large, use post-click data only
                start_time = timestamp
                end_time = timestamp + window_size/2
        else:
            # For baseline words or when not excluding, use symmetric window
            start_time = timestamp - window_size/2
            end_time = timestamp + window_size/2
        
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        window_indices = np.where(mask)[0]
        
        if len(window_indices) < 10:  # Need minimum samples
            return None
        
        # EEG features
        eeg_window = self.eeg_processed[window_indices]
        
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
                features.extend([0] * 7)  # Placeholder for frequency features
        
        # fNIRS features (if available)
        if self.fnirs is not None and hasattr(self, 'fnirs_norm'):
            fnirs_window = self.fnirs_norm[window_indices]
            
            for ch in range(fnirs_window.shape[1]):
                channel_data = fnirs_window[:, ch]
                
                # Basic statistics
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.max(channel_data) - np.min(channel_data),  # Range
                ])
                
                # Slope (rate of change) - important for hemodynamic response
                if len(channel_data) > 1:
                    time_vector = np.arange(len(channel_data))
                    slope, _ = np.polyfit(time_vector, channel_data, 1)
                    features.append(slope)
                else:
                    features.append(0)
                
                # Time to peak (simplified)
                peak_idx = np.argmax(channel_data)
                time_to_peak = peak_idx / self.sample_rate
                features.append(time_to_peak)
        
        # Inter-channel connectivity
        if eeg_window.shape[1] == 4:
            # Frontal asymmetry
            frontal_alpha_left = self._get_band_power(eeg_window[:, 1], 'alpha')  # AF7
            frontal_alpha_right = self._get_band_power(eeg_window[:, 2], 'alpha')  # AF8
            frontal_asymmetry = (frontal_alpha_right - frontal_alpha_left) / (frontal_alpha_right + frontal_alpha_left + 1e-10)
            features.append(frontal_asymmetry)
            
            # Phase synchronization between channels
            for i in range(eeg_window.shape[1]):
                for j in range(i+1, eeg_window.shape[1]):
                    sync = self._phase_sync(eeg_window[:, i], eeg_window[:, j])
                    features.append(sync)
        
        # Word-specific features
        word_features = [
            len(word),  # Word length
            self._count_syllables(word),  # Syllable count (simplified)
            1 if any(c.isdigit() for c in word) else 0,  # Contains number
            1 if word in self.confused_words else 0,  # Previously confused
            len(self.confused_words.get(word, [])),  # Times confused
        ]
        features.extend(word_features)
        
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
        """Calculate phase synchronization between two signals"""
        # Simplified phase sync using correlation of instantaneous phases
        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Phase locking value (simplified)
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
    
    def create_word_dataset(self, window_size=2.0, include_baseline_words=True):
        """Create dataset for word-level confusion detection"""
        print(f"\nCreating word-level dataset...")
        print(f"  Window size: {window_size}s")
        if self.exclude_preclick_time > 0:
            print(f"  Excluding {self.exclude_preclick_time}s before confusion clicks")
        
        X = []
        y = []
        words = []
        timestamps_list = []
        
        # Process confused words (positive samples)
        print("\nProcessing confused words...")
        confused_count = 0
        
        for word, events in self.confused_words.items():
            for timestamp, conf_type in events:
                features = self.extract_word_features(word, timestamp, window_size, is_confusion_event=True)
                if features is not None:
                    X.append(features)
                    # Label: 0=baseline, 1=word_confusion, 2=sentence_confusion
                    y.append(1 if conf_type == 'word' else 2)
                    words.append(word)
                    timestamps_list.append(timestamp)
                    confused_count += 1
        
        print(f"  Confused words processed: {confused_count}")
        
        # Process baseline words (negative samples)
        if include_baseline_words and self.all_words_timeline:
            print("\nProcessing baseline words...")
            
            # Get all confused timestamps with safety margin
            confused_times = []
            for events in self.confused_words.values():
                confused_times.extend([t for t, _ in events])
            confused_times = np.array(confused_times)
            
            baseline_count = 0
            words_seen = set()
            
            # Sample baseline words
            for timestamp, word in self.all_words_timeline:
                # Skip if too close to any confusion event
                if len(confused_times) > 0:
                    min_distance = np.min(np.abs(confused_times - timestamp))
                    if min_distance < 3.0:  # 3 second safety margin
                        continue
                
                # Skip if already seen this word many times (to avoid over-representation)
                word_key = f"{word}_{int(timestamp/10)}"  # Group by 10s windows
                if word_key in words_seen:
                    continue
                words_seen.add(word_key)
                
                # Skip very short words
                if len(word) < 3:
                    continue
                
                features = self.extract_word_features(word, timestamp, window_size, is_confusion_event=False)
                if features is not None:
                    X.append(features)
                    y.append(0)  # baseline
                    words.append(word)
                    timestamps_list.append(timestamp)
                    baseline_count += 1
                    
                    # Limit baseline samples
                    if baseline_count >= confused_count * 2:
                        break
            
            print(f"  Baseline words processed: {baseline_count}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for cls, count in zip(unique, counts):
            class_name = ['Baseline', 'Word Confusion', 'Sentence Confusion'][cls]
            print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y, words, timestamps_list
    
    def train_word_model(self, X, y, words):
        """Train model to predict word-level confusion"""
        print(f"\n{'='*40}")
        print("TRAINING WORD-LEVEL MODEL")
        if self.exclude_preclick_time > 0:
            print(f"Pre-click exclusion: {self.exclude_preclick_time}s")
        print(f"{'='*40}")
        
        # Split data
        X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(
            X, y, words, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\nTrain/Test split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        print(f"\nTraining XGBoost classifier...")
        
        # Calculate class weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=3
        )
        
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Baseline', 'Word Conf', 'Sent Conf']))
        
        # Feature importance
        feature_names = self._get_feature_names()
        importances = clf.feature_importances_
        
        # Top features
        indices = np.argsort(importances)[::-1][:20]
        
        print(f"\nTop 20 Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return {
            'classifier': clf,
            'scaler': scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'words_test': words_test,
            'feature_importances': importances,
            'feature_names': feature_names,
            'exclude_preclick_time': self.exclude_preclick_time
        }
    
    def _get_feature_names(self):
        """Generate feature names"""
        names = []
        
        # EEG features per channel
        for ch in self.channels:
            names.extend([
                f'EEG_{ch}_mean', f'EEG_{ch}_std', f'EEG_{ch}_max_abs',
                f'EEG_{ch}_skew', f'EEG_{ch}_kurtosis',
                f'EEG_{ch}_delta', f'EEG_{ch}_theta', f'EEG_{ch}_alpha',
                f'EEG_{ch}_beta', f'EEG_{ch}_gamma',
                f'EEG_{ch}_peak_freq', f'EEG_{ch}_spectral_entropy'
            ])
        
        # fNIRS features
        if self.fnirs is not None:
            for i in range(4):
                names.extend([
                    f'fNIRS_Ch{i+1}_mean', f'fNIRS_Ch{i+1}_std', 
                    f'fNIRS_Ch{i+1}_range', f'fNIRS_Ch{i+1}_slope',
                    f'fNIRS_Ch{i+1}_time_to_peak'
                ])
        
        # Connectivity features
        names.extend([
            'Frontal_asymmetry_alpha',
            'Phase_sync_TP9_AF7', 'Phase_sync_TP9_AF8', 'Phase_sync_TP9_TP10',
            'Phase_sync_AF7_AF8', 'Phase_sync_AF7_TP10', 'Phase_sync_AF8_TP10'
        ])
        
        # Word features
        names.extend([
            'Word_length', 'Word_syllables', 'Word_has_number',
            'Word_previously_confused', 'Word_confusion_count'
        ])
        
        return names
    
    def compare_models(self, results_list):
        """Compare models with different pre-click exclusion times"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        # Create comparison table
        print("Pre-click    Accuracy    Precision    Recall    F1-Score    ROC AUC")
        print("Exclusion    (Binary)    (Binary)     (Binary)  (Binary)    (Binary)")
        print("-" * 70)
        
        for results in results_list:
            # Binary metrics (any confusion vs baseline)
            y_binary_true = (results['y_test'] > 0).astype(int)
            y_binary_pred = (results['y_pred'] > 0).astype(int)
            y_binary_pred_proba = results['y_pred_proba'][:, 1] + results['y_pred_proba'][:, 2]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(results['y_test'], results['y_pred'])
            precision = precision_score(y_binary_true, y_binary_pred)
            recall = recall_score(y_binary_true, y_binary_pred)
            f1 = f1_score(y_binary_true, y_binary_pred)
            
            fpr, tpr, _ = roc_curve(y_binary_true, y_binary_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            exclusion_time = results['exclude_preclick_time']
            print(f"{exclusion_time:6.1f}s      {accuracy:8.3f}    {precision:8.3f}     "
                  f"{recall:8.3f}  {f1:8.3f}    {roc_auc:8.3f}")
        
        # Performance drop analysis
        if len(results_list) >= 2:
            original_acc = accuracy_score(results_list[0]['y_test'], results_list[0]['y_pred'])
            excluded_acc = accuracy_score(results_list[1]['y_test'], results_list[1]['y_pred'])
            
            drop = (original_acc - excluded_acc) / original_acc * 100
            
            print(f"\nPerformance drop with {results_list[1]['exclude_preclick_time']}s exclusion: {drop:.1f}%")
            
            if drop > 20:
                print("\n⚠️  SIGNIFICANT PERFORMANCE DROP DETECTED!")
                print("The model may be detecting click preparation rather than confusion.")
            elif drop > 10:
                print("\n⚠️  Moderate performance drop detected.")
                print("Some signal may come from click preparation, but confusion signal remains.")
            else:
                print("\n✓ Minimal performance drop.")
                print("The model appears to be detecting genuine confusion, not just click intent.")


def main():
    """Main function to run comparison"""
    import sys
    import glob
    import os
    
    # Find NPZ files
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for NPZ files with click data
        search_paths = [
            "*confusion_clicks*.npz",
            "~/Downloads/*confusion_clicks*.npz",
            os.path.expanduser("~/Downloads/*confusion_clicks*.npz")
        ]
        
        npz_files = []
        for path in search_paths:
            npz_files.extend(glob.glob(path))
        
        if not npz_files:
            print("No confusion click NPZ files found. Please specify a file path.")
            return
        
        print("Available NPZ files:")
        for i, f in enumerate(npz_files):
            size = os.path.getsize(f) / 1024
            print(f"  {i+1}. {os.path.basename(f)} ({size:.1f} KB)")
        
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
    
    results_list = []
    
    # Train models with different exclusion times
    exclusion_times = [0.0, 0.5, 1.0]  # No exclusion, 0.5s, 1.0s
    
    for exclude_time in exclusion_times:
        print(f"\n{'='*60}")
        print(f"Training model with {exclude_time}s pre-click exclusion")
        print(f"{'='*60}")
        
        # Create detector
        detector = WordConfusionDetector(filepath, exclude_preclick_time=exclude_time)
        
        # Create dataset
        X, y, words, timestamps = detector.create_word_dataset(
            window_size=2.0,
            include_baseline_words=True
        )
        
        if len(X) == 0:
            print("\nNo data extracted.")
            continue
        
        # Train model
        results = detector.train_word_model(X, y, words)
        results_list.append(results)
    
    # Compare results
    if len(results_list) >= 2:
        detector.compare_models(results_list)
        
        # Create comparison visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Accuracy comparison
        ax1 = plt.subplot(2, 2, 1)
        exclusion_times_used = [r['exclude_preclick_time'] for r in results_list]
        accuracies = []
        for r in results_list:
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(r['y_test'], r['y_pred'])
            accuracies.append(acc)
        
        ax1.plot(exclusion_times_used, accuracies, 'o-', markersize=10, linewidth=2)
        ax1.set_xlabel('Pre-click Exclusion Time (s)')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Model Accuracy vs Pre-click Exclusion')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot 2: Feature importance changes
        ax2 = plt.subplot(2, 2, 2)
        
        # Get top features from original model
        original_importances = results_list[0]['feature_importances']
        top_indices = np.argsort(original_importances)[::-1][:10]
        
        # Compare importances
        width = 0.35
        x = np.arange(10)
        
        for i, (results, label) in enumerate([(results_list[0], 'No exclusion'), 
                                               (results_list[-1], f'{exclusion_times[-1]}s exclusion')]):
            importances = results['feature_importances'][top_indices]
            ax2.bar(x + i*width, importances, width, label=label, alpha=0.8)
        
        ax2.set_xlabel('Top 10 Features (from original model)')
        ax2.set_ylabel('Feature Importance')
        ax2.set_title('Feature Importance Comparison')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels([results_list[0]['feature_names'][i] for i in top_indices], 
                           rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Confusion matrices side by side
        for i, (results, title) in enumerate([(results_list[0], 'No Exclusion'), 
                                              (results_list[-1], f'{exclusion_times[-1]}s Exclusion')]):
            ax = plt.subplot(2, 2, 3 + i)
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['Base', 'Word', 'Sent'], rotation=45)
            ax.set_yticklabels(['Base', 'Word', 'Sent'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {title}')
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]),
                           ha="center", va="center", 
                           color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.tight_layout()
        plt.show()
        
        print("\nANALYSIS COMPLETE")
        print("="*60)
        print("\nKey Insights:")
        print("• If performance drops significantly with pre-click exclusion,")
        print("  the model may be detecting motor preparation for clicking")
        print("• If performance remains stable, the model is likely detecting")
        print("  genuine neural signatures of confusion")
        print("\nRecommendation: Use the model with 0.5s pre-click exclusion")
        print("for more robust confusion detection in real applications.")

if __name__ == "__main__":
    main()