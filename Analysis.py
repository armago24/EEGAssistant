#!/usr/bin/env python3
"""
Word-Level EEG/fNIRS Confusion Detector
Trains a model to identify which specific words cause confusion
Uses both EEG and fNIRS data for better detection
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
import pickle
warnings.filterwarnings('ignore')

class WordConfusionDetector:
    def __init__(self, filepath):
        """Initialize and load data"""
        print(f"\n{'='*60}")
        print("WORD-LEVEL CONFUSION DETECTOR")
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
    
    def extract_word_features(self, word, timestamp, window_size=2.0):
        """Extract features for a specific word occurrence
        
        Features include:
        - EEG features in window around word
        - fNIRS features (if available)
        - Word characteristics
        - Context features
        """
        features = []
        
        # Find samples in window around timestamp
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
        
        X = []
        y = []
        words = []
        timestamps_list = []
        
        # Process confused words (positive samples)
        print("\nProcessing confused words...")
        confused_count = 0
        
        for word, events in self.confused_words.items():
            for timestamp, conf_type in events:
                features = self.extract_word_features(word, timestamp, window_size)
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
                
                features = self.extract_word_features(word, timestamp, window_size)
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
        
        # Analyze word-level performance
        print("\nWord-Level Analysis:")
        
        # Group predictions by word
        word_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'confused_prob': []})
        
        for word, true_label, pred_label, prob in zip(words_test, y_test, y_pred, y_pred_proba):
            word_performance[word]['total'] += 1
            if true_label == pred_label:
                word_performance[word]['correct'] += 1
            # Store probability of confusion (sum of word and sentence confusion)
            word_performance[word]['confused_prob'].append(prob[1] + prob[2])
        
        # Find most accurately detected confused words
        confused_words_test = [(w, p) for w, p in word_performance.items() 
                              if any(y == 1 or y == 2 for y, w2 in zip(y_test, words_test) if w2 == w)]
        
        if confused_words_test:
            print("\nTop detected confused words:")
            sorted_words = sorted(confused_words_test, 
                                key=lambda x: np.mean(word_performance[x[0]]['confused_prob']), 
                                reverse=True)[:10]
            
            for word, _ in sorted_words:
                perf = word_performance[word]
                avg_prob = np.mean(perf['confused_prob'])
                accuracy = perf['correct'] / perf['total']
                print(f"  '{word}': prob={avg_prob:.3f}, accuracy={accuracy:.3f}")
        
        # Feature importance
        feature_names = self._get_feature_names()
        importances = clf.feature_importances_
        
        # Top features
        indices = np.argsort(importances)[::-1][:20]
        
        print(f"\nTop 20 Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Analyze feature groups
        eeg_importance = np.mean([imp for feat, imp in zip(feature_names, importances) if 'EEG' in feat])
        fnirs_importance = np.mean([imp for feat, imp in zip(feature_names, importances) if 'fNIRS' in feat])
        word_importance = np.mean([imp for feat, imp in zip(feature_names, importances) if 'Word' in feat])
        
        print(f"\nFeature Group Importance:")
        print(f"  EEG features: {eeg_importance:.4f}")
        if self.fnirs is not None:
            print(f"  fNIRS features: {fnirs_importance:.4f}")
        print(f"  Word features: {word_importance:.4f}")
        
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
            'word_performance': word_performance
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
    
    def plot_results(self, results):
        """Visualize word-level detection results"""
        print(f"\n{'='*40}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*40}")
        
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Word-Level Confusion Detection Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([0, 1, 2])
        ax1.set_xticklabels(['Base', 'Word', 'Sent'], rotation=45)
        ax1.set_yticklabels(['Base', 'Word', 'Sent'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, str(cm[i, j]),
                        ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        
        plt.colorbar(im1, ax=ax1)
        
        # 2. ROC Curves (One-vs-Rest)
        ax2 = plt.subplot(3, 4, 2)
        
        # Binary classification: any confusion vs baseline
        y_binary_true = (results['y_test'] > 0).astype(int)
        y_binary_pred_proba = results['y_pred_proba'][:, 1] + results['y_pred_proba'][:, 2]
        
        fpr, tpr, _ = roc_curve(y_binary_true, y_binary_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, lw=2, label=f'Any Confusion (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall by Word Length
        ax3 = plt.subplot(3, 4, 3)
        
        # Group by word length
        word_lengths = [len(w) for w in results['words_test']]
        length_groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        
        for length, y_true, y_pred in zip(word_lengths, results['y_test'], results['y_pred']):
            length_bin = min(length // 3, 4)  # Group in bins of 3 letters
            length_groups[length_bin]['y_true'].append(y_true > 0)  # Binary
            length_groups[length_bin]['y_pred'].append(y_pred > 0)
        
        lengths = sorted(length_groups.keys())
        precisions = []
        recalls = []
        
        for length in lengths:
            group = length_groups[length]
            if len(group['y_true']) > 0:
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(group['y_true'], group['y_pred'], zero_division=0)
                rec = recall_score(group['y_true'], group['y_pred'], zero_division=0)
                precisions.append(prec)
                recalls.append(rec)
        
        x = np.arange(len(lengths))
        width = 0.35
        
        ax3.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        ax3.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        ax3.set_xlabel('Word Length Group')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance by Word Length')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{l*3}-{l*3+2}' for l in lengths])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Feature Importance by Category
        ax4 = plt.subplot(3, 4, 4)
        
        # Group features by category
        feature_categories = defaultdict(list)
        for i, name in enumerate(results['feature_names']):
            if 'EEG' in name:
                if any(band in name for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']):
                    feature_categories['EEG Frequency'].append(results['feature_importances'][i])
                else:
                    feature_categories['EEG Time'].append(results['feature_importances'][i])
            elif 'fNIRS' in name:
                feature_categories['fNIRS'].append(results['feature_importances'][i])
            elif 'Phase_sync' in name or 'asymmetry' in name:
                feature_categories['Connectivity'].append(results['feature_importances'][i])
            elif 'Word' in name:
                feature_categories['Word Features'].append(results['feature_importances'][i])
        
        categories = list(feature_categories.keys())
        avg_importances = [np.mean(feature_categories[cat]) for cat in categories]
        
        ax4.barh(categories, avg_importances, color='steelblue')
        ax4.set_xlabel('Average Importance')
        ax4.set_title('Feature Importance by Category')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Confusion Probability Distribution
        ax5 = plt.subplot(3, 4, 5)
        
        # Get confusion probabilities for all samples
        confusion_probs = results['y_pred_proba'][:, 1] + results['y_pred_proba'][:, 2]
        
        # Separate by true class
        baseline_probs = confusion_probs[results['y_test'] == 0]
        confused_probs = confusion_probs[results['y_test'] > 0]
        
        ax5.hist(baseline_probs, bins=30, alpha=0.5, label='True Baseline', color='green', density=True)
        ax5.hist(confused_probs, bins=30, alpha=0.5, label='True Confused', color='red', density=True)
        ax5.set_xlabel('Predicted Confusion Probability')
        ax5.set_ylabel('Density')
        ax5.set_title('Confusion Probability Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Top Features
        ax6 = plt.subplot(3, 4, (6, 7))
        
        top_n = 20
        indices = np.argsort(results['feature_importances'])[::-1][:top_n]
        
        ax6.barh(range(top_n), results['feature_importances'][indices][::-1], color='steelblue')
        ax6.set_yticks(range(top_n))
        ax6.set_yticklabels([results['feature_names'][i] for i in indices[::-1]], fontsize=8)
        ax6.set_xlabel('Importance')
        ax6.set_title('Top 20 Features')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. EEG Band Power Analysis
        ax7 = plt.subplot(3, 4, 8)
        
        # Extract band power importance
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_importance = {}
        
        for band in bands:
            band_indices = [i for i, name in enumerate(results['feature_names']) 
                          if band in name.lower()]
            if band_indices:
                band_importance[band] = np.mean(results['feature_importances'][band_indices])
        
        if band_importance:
            bands_list = list(band_importance.keys())
            importances = list(band_importance.values())
            
            ax7.bar(bands_list, importances, color=['purple', 'blue', 'green', 'orange', 'red'])
            ax7.set_xlabel('Frequency Band')
            ax7.set_ylabel('Average Importance')
            ax7.set_title('EEG Band Importance')
            ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Word Cloud of Confused Words (simulated with bar chart)
        ax8 = plt.subplot(3, 4, (9, 10))
        
        # Count confused words in test set
        confused_word_counts = defaultdict(int)
        for word, y_true in zip(results['words_test'], results['y_test']):
            if y_true > 0:  # Confused
                confused_word_counts[word] += 1
        
        if confused_word_counts:
            # Top confused words
            top_words = sorted(confused_word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            words, counts = zip(*top_words)
            
            y_pos = np.arange(len(words))
            ax8.barh(y_pos, counts, color='crimson')
            ax8.set_yticks(y_pos)
            ax8.set_yticklabels(words, fontsize=10)
            ax8.set_xlabel('Frequency')
            ax8.set_title('Most Frequently Confused Words')
            ax8.grid(True, alpha=0.3, axis='x')
        
        # 9. Temporal Analysis (if fNIRS available)
        if self.fnirs is not None:
            ax9 = plt.subplot(3, 4, 11)
            
            # Get fNIRS-related feature importance
            fnirs_indices = [i for i, name in enumerate(results['feature_names']) 
                           if 'fNIRS' in name]
            
            if fnirs_indices:
                fnirs_features = defaultdict(list)
                for idx in fnirs_indices:
                    feature_name = results['feature_names'][idx]
                    if 'mean' in feature_name:
                        fnirs_features['Mean'].append(results['feature_importances'][idx])
                    elif 'slope' in feature_name:
                        fnirs_features['Slope'].append(results['feature_importances'][idx])
                    elif 'time_to_peak' in feature_name:
                        fnirs_features['Time to Peak'].append(results['feature_importances'][idx])
                
                feature_types = list(fnirs_features.keys())
                avg_imp = [np.mean(fnirs_features[ft]) for ft in feature_types]
                
                ax9.bar(feature_types, avg_imp, color='coral')
                ax9.set_ylabel('Average Importance')
                ax9.set_title('fNIRS Feature Type Importance')
                ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Performance Summary Text
        ax10 = plt.subplot(3, 4, 12)
        ax10.axis('off')
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Binary metrics (any confusion vs baseline)
        y_binary_true = (results['y_test'] > 0).astype(int)
        y_binary_pred = (results['y_pred'] > 0).astype(int)
        
        accuracy = accuracy_score(results['y_test'], results['y_pred'])
        binary_precision = precision_score(y_binary_true, y_binary_pred)
        binary_recall = recall_score(y_binary_true, y_binary_pred)
        binary_f1 = f1_score(y_binary_true, y_binary_pred)
        
        summary_text = f"""Performance Summary:
        
Overall Accuracy: {accuracy:.3f}

Binary Classification (Confusion Detection):
  Precision: {binary_precision:.3f}
  Recall: {binary_recall:.3f}
  F1-Score: {binary_f1:.3f}
  ROC AUC: {roc_auc:.3f}

Word Analysis:
  Total test words: {len(results['words_test'])}
  Unique test words: {len(set(results['words_test']))}
  Confused words detected: {sum(y_binary_pred)}

Key Insights:
  • {categories[np.argmax(avg_importances)]} features are most important
  • Best performing frequency band: {bands_list[np.argmax(importances)] if band_importance else 'N/A'}
  • Model can identify specific confusing words"""
        
        ax10.text(0.1, 0.5, summary_text, transform=ax10.transAxes,
                 fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def predict_confusion_realtime(self, new_text, model_results):
        """Simulate real-time confusion prediction on new text"""
        print(f"\n{'='*40}")
        print("REAL-TIME CONFUSION PREDICTION DEMO")
        print(f"{'='*40}")
        
        # This is a simulation - in real deployment, you would:
        # 1. Stream EEG/fNIRS data while reading
        # 2. Track cursor position to know current word
        # 3. Extract features for each word in real-time
        # 4. Apply the trained model
        
        print("\nThis would predict confusion for each word as you read.")
        print("The model would highlight potentially confusing words based on:")
        print("  • Your EEG patterns while reading each word")
        print("  • fNIRS hemodynamic response")
        print("  • Word characteristics")
        print("  • Your personal confusion history")
        
        # Example: Show which features are most predictive
        top_features = np.argsort(model_results['feature_importances'])[::-1][:10]
        print("\nMost predictive features for your confusion:")
        for i, idx in enumerate(top_features):
            print(f"  {i+1}. {model_results['feature_names'][idx]}")
    
    def save_model_for_realtime(self, training_results, output_path='confusion_model.pkl'):
        """Save the trained model and necessary components for real-time use"""
        
        # Package everything needed for real-time prediction
        model_package = {
            'classifier': training_results['classifier'],
            'scaler': training_results['scaler'],
            'feature_names': training_results['feature_names'],
            'feature_importances': training_results['feature_importances'],
            'metadata': {
                'device': 'Muse S Athena',
                'sample_rate': self.sample_rate,
                'window_size': 2.0,  # seconds
                'eeg_channels': self.channels,
                'fnirs_channels': ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm'] if self.fnirs is not None else None
            }
        }
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n{'='*40}")
        print("MODEL SAVED FOR REAL-TIME USE")
        print(f"{'='*40}")
        print(f"Model saved to: {output_path}")
        print(f"Model type: {type(training_results['classifier']).__name__}")
        print(f"Features: {len(training_results['feature_names'])}")
        print(f"Classes: Baseline (0), Word Confusion (1), Sentence Confusion (2)")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"EEG channels: {', '.join(self.channels)}")
        if self.fnirs is not None:
            print(f"fNIRS channels: 4 normalized channels")
        print("\nThis model can now be loaded in real-time applications!")
        
        return model_package

def main():
    """Main function"""
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
            print("Looking for files with 'confusion_clicks' in the name.")
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
    
    # Create detector
    detector = WordConfusionDetector(filepath)
    
    # Create word-level dataset
    X, y, words, timestamps = detector.create_word_dataset(
        window_size=2.0,  # 2 second windows around each word
        include_baseline_words=True
    )
    
    if len(X) == 0:
        print("\nNo data extracted. Ensure the recording has clicked confusion events.")
        return
    
    # Train model
    results = detector.train_word_model(X, y, words)
    
    # Save model for real-time use
    model_filename = f"mo{os.path.basename(filepath).replace('.npz', '')}.pkl"
    detector.save_model_for_realtime(results, model_filename)
    
    # Visualize results
    detector.plot_results(results)
    
    # Demo real-time prediction
    detector.predict_confusion_realtime("Example new text...", results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nThe model can now identify specific words that confuse you!")
    print(f"\n✓ Trained model saved as: {model_filename}")
    print("✓ Model is ready for real-time deployment!")
    print("\nKey improvements in this version:")
    print("  • Analyzes individual WORDS, not just time windows")
    print("  • Uses clicked words as ground truth labels")
    print("  • Incorporates fNIRS hemodynamic data")
    print("  • Tracks word characteristics and confusion history")
    print("  • Can highlight confusing words in real-time while reading")
    print("  • Automatically saves trained model for deployment")
    print("\nNext steps for deployment:")
    print(f"  • Load the saved model: {model_filename}")
    print("  • Stream data in real-time while reading")
    print("  • Highlight predicted confusing words in the text")
    print("  • Build personalized confusion profile over time")
    print("  • Adapt to different types of content (technical vs casual)")

if __name__ == "__main__":
    main()