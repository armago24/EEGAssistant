#!/usr/bin/env python3
"""
xgboost.py:
XGBoost-based confusion detection from EEG/fNIRS data
Comprehensive feature engineering based on neuroscience principles
Rigorous evaluation with leave-one-session-out cross-validation

Neural correlates of confusion:
- Increased frontal theta (4-8 Hz): cognitive effort and working memory load
- Decreased alpha (8-13 Hz): reduced relaxation, increased attention
- Frontal asymmetry changes: differential hemisphere engagement
- fNIRS hemodynamic response: increased frontal oxygenation (6s delay)
- Reduced inter-channel coherence: disrupted neural synchronization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.utils import class_weight
import xgboost as xgb
from collections import defaultdict
import os
from datetime import datetime
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')


class ConfusionDetector:
    def __init__(self, window_size_sec=2.0, overlap=0.5, sample_rate=256):
        """
        Initialize confusion detector

        Args:
            window_size_sec: Window size in seconds (default 2.0s = 512 samples)
            overlap: Overlap between windows (default 0.5 = 50%)
            sample_rate: EEG sampling rate in Hz (default 256)
        """
        self.window_size_sec = window_size_sec
        self.window_size = int(window_size_sec * sample_rate)
        self.overlap = overlap
        self.step_size = int(self.window_size * (1 - overlap))
        self.sample_rate = sample_rate

        # Frequency bands aligned with cognitive neuroscience
        self.bands = {
            'delta': (0.5, 4),    # Deep sleep, unconscious processes
            'theta': (4, 8),      # Working memory, cognitive control
            'alpha': (8, 13),     # Relaxation, inhibition
            'beta': (13, 30),     # Active thinking, focus
            'gamma': (30, 50)     # Integration, binding
        }

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        # EEG channel names and positions
        self.eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        # TP9/TP10: Temporal (left/right)
        # AF7/AF8: Frontal (left/right)

    def load_data(self, npz_files):
        """Load multiple NPZ files and extract relevant data"""
        all_data = []

        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

        for i, filepath in enumerate(npz_files):
            print(f"\nFile {i+1}/{len(npz_files)}: {os.path.basename(filepath)}")

            try:
                data = np.load(filepath, allow_pickle=True)

                # Validate required arrays
                required = ['relative_timestamps', 'eeg', 'fnirs', 'motion',
                           'event_timestamps', 'event_types']
                for key in required:
                    if key not in data:
                        print(f"  WARNING: Missing '{key}' - skipping file")
                        continue

                # Convert event timestamps to relative time
                # event_timestamps are in absolute Unix time, need to convert to relative
                abs_timestamps = data['timestamps']  # absolute Unix timestamps
                rel_timestamps = data['relative_timestamps']  # relative (0 to duration)
                event_timestamps_abs = data['event_timestamps']

                # Convert events to relative time
                event_timestamps_rel = event_timestamps_abs - abs_timestamps[0]

                session_data = {
                    'timestamps': rel_timestamps,  # use relative timestamps
                    'eeg': data['eeg'],
                    'fnirs': data['fnirs'],
                    'motion': data['motion'],
                    'event_timestamps': event_timestamps_rel,  # converted to relative
                    'event_types': data['event_types'],
                    'session_id': i,
                    'filename': os.path.basename(filepath)
                }

                # Validate shapes
                n_samples = len(session_data['timestamps'])
                if session_data['eeg'].shape[0] != n_samples:
                    print(f"  WARNING: EEG shape mismatch - skipping")
                    continue

                print(f"  Samples: {n_samples:,}")
                print(f"  Duration: {session_data['timestamps'][-1]:.1f}s")
                print(f"  Events: {len(session_data['event_timestamps'])}")

                # Count event types
                word_conf = np.sum(session_data['event_types'] == 'word_confusion')
                sent_conf = np.sum(session_data['event_types'] == 'sentence_confusion')
                print(f"    Word confusion: {word_conf}")
                print(f"    Sentence confusion: {sent_conf}")

                all_data.append(session_data)

            except Exception as e:
                print(f"  ERROR loading file: {e}")
                continue

        if not all_data:
            raise ValueError("No valid data files loaded")

        print(f"\nSuccessfully loaded {len(all_data)} sessions")
        return all_data

    def extract_windows(self, session_data):
        """Extract sliding windows with confusion labels"""
        timestamps = session_data['timestamps']  # relative timestamps (0 to duration)
        eeg = session_data['eeg']
        fnirs = session_data['fnirs']
        motion = session_data['motion']
        event_timestamps = session_data['event_timestamps']
        event_types = session_data['event_types']
        session_id = session_data['session_id']

        windows = []
        labels = []
        session_ids = []
        event_info = []

        # Sliding window extraction
        for start_idx in range(0, len(timestamps) - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size

            window_start_time = timestamps[start_idx]
            window_end_time = timestamps[end_idx - 1]
            window_center_time = (window_start_time + window_end_time) / 2

            # Extract window data
            window_eeg = eeg[start_idx:end_idx]
            window_fnirs = fnirs[start_idx:end_idx]
            window_motion = motion[start_idx:end_idx]

            # Validate window
            if window_eeg.shape[0] != self.window_size:
                continue
            if np.any(np.isnan(window_eeg)) or np.any(np.isnan(window_fnirs)):
                continue

            # Determine label - check if any event falls within window
            label = 0  # baseline
            matched_event = None

            for evt_time, evt_type in zip(event_timestamps, event_types):
                # Event within window boundaries
                if window_start_time <= evt_time <= window_end_time:
                    if evt_type == 'word_confusion':
                        label = 1
                    elif evt_type == 'sentence_confusion':
                        label = 2
                    matched_event = (evt_time, evt_type)
                    break  # Use first matching event

            windows.append({
                'eeg': window_eeg,
                'fnirs': window_fnirs,
                'motion': window_motion,
                'time': window_center_time
            })
            labels.append(label)
            session_ids.append(session_id)
            event_info.append(matched_event)

        return windows, labels, session_ids, event_info

    def compute_band_power(self, data, band):
        """Compute power in frequency band using Welch's method"""
        try:
            freqs, psd = signal.welch(
                data,
                fs=self.sample_rate,
                nperseg=min(len(data), 256),
                noverlap=128,
                scaling='density'
            )

            # Find indices in band
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            if not np.any(band_mask):
                return 0.0

            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            return band_power
        except:
            return 0.0

    def compute_spectral_entropy(self, data):
        """Compute spectral entropy as measure of signal complexity"""
        try:
            freqs, psd = signal.welch(
                data,
                fs=self.sample_rate,
                nperseg=min(len(data), 256),
                noverlap=128
            )

            # Normalize PSD to get probability distribution
            psd_norm = psd / np.sum(psd)

            # Compute entropy
            spec_entropy = entropy(psd_norm)
            return spec_entropy
        except:
            return 0.0

    def extract_features_single_window(self, window):
        """Extract comprehensive feature set from a single window"""
        features = []
        feature_names = []

        eeg = window['eeg']  # [window_size, 4]
        fnirs = window['fnirs']  # [window_size, 8]
        motion = window['motion']  # [window_size, 6]

        # ==================== EEG FEATURES ====================

        # Per-channel features
        for ch_idx, ch_name in enumerate(self.eeg_channels):
            ch_data = eeg[:, ch_idx]

            # Time-domain statistical features
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                skew(ch_data),
                kurtosis(ch_data),
                np.ptp(ch_data),  # peak-to-peak range
                np.percentile(ch_data, 25),
                np.percentile(ch_data, 75),
            ])
            feature_names.extend([
                f'eeg_{ch_name}_mean',
                f'eeg_{ch_name}_std',
                f'eeg_{ch_name}_var',
                f'eeg_{ch_name}_skew',
                f'eeg_{ch_name}_kurtosis',
                f'eeg_{ch_name}_ptp',
                f'eeg_{ch_name}_q25',
                f'eeg_{ch_name}_q75',
            ])

            # Temporal dynamics
            ch_diff = np.diff(ch_data)
            features.extend([
                np.mean(ch_diff),  # first derivative (rate of change)
                np.std(ch_diff),
            ])
            feature_names.extend([
                f'eeg_{ch_name}_diff_mean',
                f'eeg_{ch_name}_diff_std',
            ])

            # Frequency-domain features
            total_power = 0
            band_powers = {}

            for band_name, band_range in self.bands.items():
                power = self.compute_band_power(ch_data, band_range)
                band_powers[band_name] = power
                total_power += power
                features.append(power)
                feature_names.append(f'eeg_{ch_name}_{band_name}_power')

            # Relative band powers (normalized)
            for band_name, power in band_powers.items():
                rel_power = power / (total_power + 1e-10)
                features.append(rel_power)
                feature_names.append(f'eeg_{ch_name}_{band_name}_rel_power')

            # Spectral entropy (complexity measure)
            spec_ent = self.compute_spectral_entropy(ch_data)
            features.append(spec_ent)
            feature_names.append(f'eeg_{ch_name}_spectral_entropy')

        # ==================== SPATIAL EEG FEATURES ====================

        # Frontal asymmetry (AF7 vs AF8) - key for emotion/cognitive processing
        af7_data = eeg[:, 1]
        af8_data = eeg[:, 2]

        for band_name, band_range in self.bands.items():
            af7_power = self.compute_band_power(af7_data, band_range)
            af8_power = self.compute_band_power(af8_data, band_range)

            # Asymmetry index: (L - R) / (L + R)
            asymmetry = (af7_power - af8_power) / (af7_power + af8_power + 1e-10)
            features.append(asymmetry)
            feature_names.append(f'frontal_asymmetry_{band_name}')

        # Temporal asymmetry (TP9 vs TP10)
        tp9_data = eeg[:, 0]
        tp10_data = eeg[:, 3]

        for band_name, band_range in self.bands.items():
            tp9_power = self.compute_band_power(tp9_data, band_range)
            tp10_power = self.compute_band_power(tp10_data, band_range)

            asymmetry = (tp9_power - tp10_power) / (tp9_power + tp10_power + 1e-10)
            features.append(asymmetry)
            feature_names.append(f'temporal_asymmetry_{band_name}')

        # Inter-channel correlations (connectivity)
        for i in range(len(self.eeg_channels)):
            for j in range(i + 1, len(self.eeg_channels)):
                corr = np.corrcoef(eeg[:, i], eeg[:, j])[0, 1]
                features.append(corr)
                feature_names.append(f'eeg_corr_{self.eeg_channels[i]}_{self.eeg_channels[j]}')

        # Global EEG features
        eeg_mean = np.mean(eeg, axis=1)  # Average across channels
        features.extend([
            np.mean(eeg_mean),
            np.std(eeg_mean),
        ])
        feature_names.extend([
            'eeg_global_mean',
            'eeg_global_std',
        ])

        # ==================== fNIRS FEATURES ====================

        # Use normalized channels (first 4 of 8)
        fnirs_norm = fnirs[:, :4]

        for ch_idx in range(4):
            ch_data = fnirs_norm[:, ch_idx]

            # Time-domain features
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.min(ch_data),
                np.max(ch_data),
            ])
            feature_names.extend([
                f'fnirs_ch{ch_idx+1}_mean',
                f'fnirs_ch{ch_idx+1}_std',
                f'fnirs_ch{ch_idx+1}_min',
                f'fnirs_ch{ch_idx+1}_max',
            ])

            # Hemodynamic trend (linear slope)
            x = np.arange(len(ch_data))
            slope = np.polyfit(x, ch_data, 1)[0]
            features.append(slope)
            feature_names.append(f'fnirs_ch{ch_idx+1}_slope')

            # First derivative (rate of change)
            ch_diff = np.diff(ch_data)
            features.extend([
                np.mean(ch_diff),
                np.std(ch_diff),
            ])
            feature_names.extend([
                f'fnirs_ch{ch_idx+1}_diff_mean',
                f'fnirs_ch{ch_idx+1}_diff_std',
            ])

        # Global fNIRS features
        features.extend([
            np.mean(fnirs_norm),
            np.std(fnirs_norm),
        ])
        feature_names.extend([
            'fnirs_global_mean',
            'fnirs_global_std',
        ])

        # ==================== MOTION FEATURES ====================

        acc = motion[:, :3]  # accelerometer
        gyro = motion[:, 3:]  # gyroscope

        # Acceleration magnitude (head movement)
        acc_mag = np.linalg.norm(acc, axis=1)
        features.extend([
            np.mean(acc_mag),
            np.std(acc_mag),
            np.max(acc_mag),
        ])
        feature_names.extend([
            'acc_magnitude_mean',
            'acc_magnitude_std',
            'acc_magnitude_max',
        ])

        # Angular velocity magnitude (head rotation)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        features.extend([
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.max(gyro_mag),
        ])
        feature_names.extend([
            'gyro_magnitude_mean',
            'gyro_magnitude_std',
            'gyro_magnitude_max',
        ])

        # Movement variability (restlessness indicator)
        features.extend([
            np.var(acc_mag),
            np.var(gyro_mag),
        ])
        feature_names.extend([
            'acc_magnitude_var',
            'gyro_magnitude_var',
        ])

        return np.array(features), feature_names

    def extract_features(self, windows):
        """Extract features from all windows with progress tracking"""
        all_features = []

        print("\nExtracting features...")
        total = len(windows)

        for i, window in enumerate(windows):
            if i % 500 == 0 or i == total - 1:
                pct = 100 * (i + 1) / total
                print(f"  Progress: {i+1:,}/{total:,} ({pct:.1f}%)")

            features, feature_names = self.extract_features_single_window(window)
            all_features.append(features)

        # Store feature names (same for all windows)
        if not self.feature_names:
            self.feature_names = feature_names

        print(f"Feature extraction complete: {len(feature_names)} features per window")
        return np.array(all_features)

    def train(self, npz_files, output_dir='xgboost_results'):
        """Train the model with comprehensive evaluation"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("CONFUSION DETECTION - XGBOOST TRAINING SYSTEM")
        print("="*80)

        # Load all sessions
        all_sessions = self.load_data(npz_files)

        # Extract windows from all sessions
        all_windows = []
        all_labels = []
        all_session_ids = []
        all_event_info = []

        print("\n" + "="*80)
        print("EXTRACTING WINDOWS")
        print("="*80)
        print(f"\nWindow size: {self.window_size_sec}s ({self.window_size} samples)")
        print(f"Overlap: {self.overlap*100:.0f}% ({int(self.window_size*self.overlap)} samples)")
        print(f"Step size: {self.step_size} samples")

        for session_data in all_sessions:
            print(f"\nProcessing {session_data['filename']}...")
            windows, labels, session_ids, event_info = self.extract_windows(session_data)

            print(f"  Extracted {len(windows)} windows")
            print(f"  Baseline: {np.sum(np.array(labels) == 0)}")
            print(f"  Word confusion: {np.sum(np.array(labels) == 1)}")
            print(f"  Sentence confusion: {np.sum(np.array(labels) == 2)}")

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_session_ids.extend(session_ids)
            all_event_info.extend(event_info)

        all_labels = np.array(all_labels)
        all_session_ids = np.array(all_session_ids)

        print("\n" + "-"*80)
        print(f"TOTAL WINDOWS: {len(all_windows):,}")
        print("-"*80)

        class_names = ['Baseline', 'Word Confusion', 'Sentence Confusion']
        for cls in [0, 1, 2]:
            count = np.sum(all_labels == cls)
            pct = 100 * count / len(all_labels)
            print(f"{class_names[cls]:20s}: {count:6,} ({pct:5.1f}%)")

        # Calculate class imbalance ratio
        baseline_count = np.sum(all_labels == 0)
        confusion_count = np.sum(all_labels > 0)
        imbalance_ratio = baseline_count / (confusion_count + 1e-10)
        print(f"\nClass imbalance ratio (baseline:confusion): {imbalance_ratio:.1f}:1")

        # Extract features
        X = self.extract_features(all_windows)
        y = all_labels
        groups = all_session_ids

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Total features: {X.shape[1]}")

        # Check for invalid features
        invalid_mask = np.any(np.isnan(X) | np.isinf(X), axis=0)
        if np.any(invalid_mask):
            print(f"\nWARNING: Found {np.sum(invalid_mask)} features with NaN/Inf values")
            print("Replacing with zeros...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Save feature names
        feature_file = os.path.join(output_dir, 'feature_names.txt')
        with open(feature_file, 'w') as f:
            f.write(f"Total features: {len(self.feature_names)}\n")
            f.write("-"*80 + "\n")
            for i, name in enumerate(self.feature_names):
                f.write(f"{i:4d}: {name}\n")
        print(f"\nFeature names saved to: {feature_file}")

        # ==================== CROSS-VALIDATION ====================
        print("\n" + "="*80)
        print("LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
        print("="*80)

        logo = LeaveOneGroupOut()
        n_folds = len(np.unique(groups))

        cv_predictions = np.zeros(len(y))
        cv_probabilities = np.zeros((len(y), 3))
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_session = np.unique(groups[test_idx])[0]
            train_sessions = np.unique(groups[train_idx])

            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*60}")
            print(f"Train sessions: {train_sessions}")
            print(f"Test session: {test_session}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"Train size: {len(X_train):,}")
            print(f"Test size: {len(X_test):,}")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Compute class weights
            classes = np.unique(y_train)
            weights = class_weight.compute_class_weight(
                'balanced',
                classes=classes,
                y=y_train
            )
            class_weights = dict(zip(classes, weights))
            sample_weights = np.array([class_weights[label] for label in y_train])

            print(f"Class weights: {class_weights}")

            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

            cv_predictions[test_idx] = y_pred
            cv_probabilities[test_idx] = y_prob

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred,
                labels=[0, 1, 2],
                average='weighted',
                zero_division=0
            )

            print(f"\nFold Results:")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall: {rec:.3f}")
            print(f"  F1: {f1:.3f}")

            fold_metrics.append({
                'fold': fold + 1,
                'test_session': test_session,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

        # ==================== FINAL MODEL ====================
        print("\n" + "="*80)
        print("TRAINING FINAL MODEL ON ALL DATA")
        print("="*80)

        # Scale all features
        X_scaled = self.scaler.fit_transform(X)

        # Compute class weights
        classes = np.unique(y)
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
        class_weights = dict(zip(classes, weights))
        sample_weights = np.array([class_weights[label] for label in y])

        print(f"Class weights: {class_weights}")

        # Train final model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        print("Training...")
        self.model.fit(X_scaled, y, sample_weight=sample_weights, verbose=True)

        # ==================== EVALUATION ====================
        self.generate_plots(X, y, groups, cv_predictions, cv_probabilities,
                          fold_metrics, output_dir)

        # ==================== SAVE MODEL ====================
        model_path = os.path.join(output_dir, 'confusion_detector.pkl')

        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'window_size_sec': self.window_size_sec,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'sample_rate': self.sample_rate,
            'bands': self.bands,
            'eeg_channels': self.eeg_channels,
            'class_names': class_names,
            'class_weights': class_weights,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)

        print(f"\nModel saved to: {model_path}")

        return cv_predictions, cv_probabilities, fold_metrics

    def generate_plots(self, X, y, groups, cv_predictions, cv_probabilities,
                      fold_metrics, output_dir):
        """Generate comprehensive evaluation plots"""

        print("\n" + "="*80)
        print("GENERATING EVALUATION PLOTS")
        print("="*80)

        # Create figure directory
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        class_names = ['Baseline', 'Word Confusion', 'Sentence Confusion']
        colors = ['#2ecc71', '#e74c3c', '#f39c12']

        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')

        # ===== Plot 1: Confusion Matrix =====
        print("\n1. Generating confusion matrix...")

        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, cv_predictions)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.title('Confusion Matrix\n(Leave-One-Session-Out Cross-Validation)',
                 fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(label='Proportion')

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                        ha="center", va="center", fontsize=12,
                        color=text_color, fontweight='bold')

        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '01_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 2: Class Distribution =====
        print("2. Generating class distribution...")

        plt.figure(figsize=(10, 6))
        class_counts = [np.sum(y == i) for i in range(3)]

        bars = plt.bar(class_names, class_counts, color=colors, alpha=0.8, edgecolor='black')
        plt.title('Training Data Class Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Number of Windows', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({100*count/len(y):.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '02_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 3: ROC Curves =====
        print("3. Generating ROC curves...")

        plt.figure(figsize=(10, 8))

        for cls in range(3):
            y_binary = (y == cls).astype(int)
            y_scores = cv_probabilities[:, cls]

            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=3, label=f'{class_names[cls]} (AUC = {roc_auc:.3f})',
                    color=colors[cls])

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '03_roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 4: Precision-Recall Curves =====
        print("4. Generating precision-recall curves...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for cls, ax in enumerate(axes):
            y_binary = (y == cls).astype(int)
            y_scores = cv_probabilities[:, cls]

            precision, recall, _ = precision_recall_curve(y_binary, y_scores)
            avg_precision = auc(recall, precision)

            ax.plot(recall, precision, lw=3, color=colors[cls],
                   label=f'AP = {avg_precision:.3f}')
            ax.fill_between(recall, precision, alpha=0.2, color=colors[cls])

            baseline = np.sum(y_binary) / len(y_binary)
            ax.axhline(y=baseline, color='red', linestyle='--', lw=2,
                      label=f'Baseline = {baseline:.3f}', alpha=0.7)

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title(class_names[cls], fontsize=14, fontweight='bold')
            ax.legend(loc="best", fontsize=11, framealpha=0.9)
            ax.grid(alpha=0.3)

        plt.suptitle('Precision-Recall Curves', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '04_precision_recall.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 5: Feature Importance =====
        print("5. Generating feature importance...")

        if self.model is not None:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:25]  # Top 25

            plt.figure(figsize=(12, 10))
            plt.barh(range(25), importances[indices], color='steelblue',
                    alpha=0.8, edgecolor='black')
            plt.yticks(range(25), [self.feature_names[i] for i in indices], fontsize=11)
            plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
            plt.title('Top 25 Most Important Features', fontsize=16, fontweight='bold', pad=20)
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, '05_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Save full feature importance
            importance_file = os.path.join(output_dir, 'feature_importance.txt')
            with open(importance_file, 'w') as f:
                f.write("Feature Importance Rankings\n")
                f.write("="*80 + "\n")
                sorted_idx = np.argsort(importances)[::-1]
                for rank, idx in enumerate(sorted_idx, 1):
                    f.write(f"{rank:3d}. {self.feature_names[idx]:<50s} {importances[idx]:.6f}\n")

            print(f"   Full feature importance saved to: {importance_file}")

        # ===== Plot 6: Session Performance =====
        print("6. Generating session performance...")

        unique_sessions = np.unique(groups)
        session_metrics = {s: {'acc': [], 'f1': []} for s in unique_sessions}

        for metric in fold_metrics:
            session_metrics[metric['test_session']]['acc'].append(metric['accuracy'])
            session_metrics[metric['test_session']]['f1'].append(metric['f1'])

        session_accuracies = [session_metrics[s]['acc'][0] for s in unique_sessions]
        session_f1_scores = [session_metrics[s]['f1'][0] for s in unique_sessions]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x_pos = np.arange(len(unique_sessions))

        ax1.bar(x_pos, session_accuracies, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Session ID', fontsize=14, fontweight='bold')
        ax1.set_title('Per-Session Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{s}' for s in unique_sessions])
        mean_acc = np.mean(session_accuracies)
        ax1.axhline(y=mean_acc, color='red', linestyle='--', lw=2,
                   label=f'Mean = {mean_acc:.3f}', alpha=0.7)
        ax1.legend(fontsize=12)
        ax1.grid(alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])

        ax2.bar(x_pos, session_f1_scores, color='darkorange', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Weighted F1 Score', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Session ID', fontsize=14, fontweight='bold')
        ax2.set_title('Per-Session F1 Score', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{s}' for s in unique_sessions])
        mean_f1 = np.mean(session_f1_scores)
        ax2.axhline(y=mean_f1, color='red', linestyle='--', lw=2,
                   label=f'Mean = {mean_f1:.3f}', alpha=0.7)
        ax2.legend(fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])

        plt.suptitle('Session-wise Performance', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '06_session_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 7: Confidence Distribution =====
        print("7. Generating prediction confidence distribution...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for cls, ax in enumerate(axes):
            class_mask = y == cls
            correct_mask = cv_predictions == cls

            correct_conf = cv_probabilities[class_mask & correct_mask, cls]
            incorrect_conf = cv_probabilities[class_mask & ~correct_mask, cls]

            bins = np.linspace(0, 1, 21)
            ax.hist(correct_conf, bins=bins, alpha=0.7, color='green',
                   label=f'Correct (n={len(correct_conf):,})', edgecolor='black')
            ax.hist(incorrect_conf, bins=bins, alpha=0.7, color='red',
                   label=f'Incorrect (n={len(incorrect_conf):,})', edgecolor='black')

            ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'{class_names[cls]}\n(True Labels)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11, framealpha=0.9)
            ax.grid(alpha=0.3, axis='y')
            ax.set_xlim([0, 1])

        plt.suptitle('Prediction Confidence Distribution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '07_confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Plot 8: Metrics Summary =====
        print("8. Generating metrics summary...")

        precision, recall, f1, support = precision_recall_fscore_support(
            y, cv_predictions, labels=[0, 1, 2], zero_division=0
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(class_names))
        width = 0.25

        ax.bar(x - width, precision, width, label='Precision',
              color='steelblue', alpha=0.9, edgecolor='black')
        ax.bar(x, recall, width, label='Recall',
              color='darkorange', alpha=0.9, edgecolor='black')
        ax.bar(x + width, f1, width, label='F1 Score',
              color='green', alpha=0.9, edgecolor='black')

        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Classification Metrics by Class', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=12)
        ax.legend(fontsize=13, framealpha=0.9)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        # Add value labels
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.03, f'{p:.2f}', ha='center', fontsize=11, fontweight='bold')
            ax.text(i, r + 0.03, f'{r:.2f}', ha='center', fontsize=11, fontweight='bold')
            ax.text(i + width, f + 0.03, f'{f:.2f}', ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '08_metrics_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ===== Generate Text Report =====
        print("9. Generating evaluation report...")

        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONFUSION DETECTION MODEL - EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: XGBoost Classifier\n")
            f.write(f"Validation: Leave-One-Session-Out Cross-Validation\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("DATASET SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total windows: {len(y):,}\n")
            f.write(f"Window size: {self.window_size_sec}s ({self.window_size} samples)\n")
            f.write(f"Overlap: {self.overlap * 100:.0f}%\n")
            f.write(f"Step size: {self.step_size} samples\n")
            f.write(f"Number of sessions: {len(np.unique(groups))}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")

            f.write("\nClass Distribution:\n")
            for cls, name in enumerate(class_names):
                count = np.sum(y == cls)
                pct = 100 * count / len(y)
                f.write(f"  {name:20s}: {count:6,} ({pct:5.1f}%)\n")

            baseline_count = np.sum(y == 0)
            confusion_count = np.sum(y > 0)
            imbalance = baseline_count / (confusion_count + 1e-10)
            f.write(f"\nClass Imbalance Ratio: {imbalance:.1f}:1 (baseline:confusion)\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("OVERALL PERFORMANCE (CROSS-VALIDATION)\n")
            f.write("-"*80 + "\n")

            overall_acc = accuracy_score(y, cv_predictions)
            f.write(f"Overall Accuracy: {overall_acc:.4f}\n")

            f.write("\nPer-Class Metrics:\n")
            f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-"*80 + "\n")

            for cls, name in enumerate(class_names):
                f.write(f"{name:<25} {precision[cls]:<12.4f} {recall[cls]:<12.4f} "
                       f"{f1[cls]:<12.4f} {support[cls]:<10}\n")

            weighted_p = np.average(precision, weights=support)
            weighted_r = np.average(recall, weights=support)
            weighted_f1 = np.average(f1, weights=support)

            f.write("-"*80 + "\n")
            f.write(f"{'Weighted Average':<25} {weighted_p:<12.4f} {weighted_r:<12.4f} "
                   f"{weighted_f1:<12.4f} {np.sum(support):<10}\n")

            f.write("\nConfusion Matrix:\n")
            header = "True \\ Pred"
            f.write(f"{header:<15}")
            for name in class_names:
                f.write(f"{name:<15}")
            f.write("\n")
            f.write("-"*80 + "\n")
            for i, name in enumerate(class_names):
                f.write(f"{name:<15}")
                for j in range(len(class_names)):
                    f.write(f"{cm[i, j]:<15}")
                f.write("\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("SESSION-WISE PERFORMANCE\n")
            f.write("-"*80 + "\n")

            for metric in fold_metrics:
                f.write(f"Session {metric['test_session']}: ")
                f.write(f"Acc={metric['accuracy']:.4f}, ")
                f.write(f"Prec={metric['precision']:.4f}, ")
                f.write(f"Rec={metric['recall']:.4f}, ")
                f.write(f"F1={metric['f1']:.4f}\n")

            f.write(f"\nMean Accuracy: {np.mean(session_accuracies):.4f} ")
            f.write(f"(std={np.std(session_accuracies):.4f})\n")
            f.write(f"Mean F1 Score: {np.mean(session_f1_scores):.4f} ")
            f.write(f"(std={np.std(session_f1_scores):.4f})\n")

            f.write("\n" + "-"*80 + "\n")
            f.write("TOP 15 MOST IMPORTANT FEATURES\n")
            f.write("-"*80 + "\n")

            if self.model is not None:
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]

                for rank, idx in enumerate(indices, 1):
                    f.write(f"{rank:3d}. {self.feature_names[idx]:<50s} ")
                    f.write(f"{importances[idx]:.6f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION AND RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            f.write("\n")
            f.write("1. CLASS IMBALANCE:\n")
            f.write("   The dataset has severe class imbalance favoring baseline states.\n")
            f.write("   Class weights were applied during training to mitigate this.\n")
            f.write("   Consider collecting more confusion events in future sessions.\n")
            f.write("\n")
            f.write("2. CROSS-VALIDATION STRATEGY:\n")
            f.write("   Leave-one-session-out CV tests generalization to NEW recording\n")
            f.write("   sessions. Performance variance across sessions indicates individual\n")
            f.write("   differences or context-dependent patterns.\n")
            f.write("\n")
            f.write("3. EXPECTED LIVE PERFORMANCE:\n")
            f.write("   Real-time detection may show lower accuracy than reported here\n")
            f.write("   due to temporal dynamics and online processing constraints.\n")
            f.write("   Recommend confidence thresholding and temporal smoothing.\n")
            f.write("\n")
            f.write("4. FEATURE ANALYSIS:\n")
            f.write("   Top features indicate which brain signals most strongly predict\n")
            f.write("   confusion. Frequency band powers and spatial asymmetries are\n")
            f.write("   typically most informative for cognitive load detection.\n")
            f.write("\n")
            f.write("5. NEXT STEPS:\n")
            f.write("   - Collect 5-10 more training sessions for better generalization\n")
            f.write("   - Implement temporal smoothing in live detection\n")
            f.write("   - Add confidence thresholding (e.g., require >0.7 probability)\n")
            f.write("   - Consider ensemble methods or session-specific calibration\n")
            f.write("\n")

        print(f"\n   Report saved to: {report_path}")
        print(f"   All figures saved to: {fig_dir}")
        print("\nPlot generation complete!")


def main():
    """Main training function"""

    # Find training data
    data_dir = 'fixed_tdata'
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory '{data_dir}' not found")
        print("Please ensure your training data is in the fixed_tdata/ directory")
        return

    npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    if not npz_files:
        print(f"ERROR: No NPZ files found in {data_dir}/")
        print("Please add your training data files to this directory")
        return

    print("\n" + "="*80)
    print("CONFUSION DETECTION - XGBOOST TRAINING SYSTEM")
    print("="*80)
    print(f"\nFound {len(npz_files)} training files in {data_dir}/:")
    for i, f in enumerate(npz_files, 1):
        print(f"  {i}. {os.path.basename(f)}")

    # Initialize detector with optimal parameters
    detector = ConfusionDetector(
        window_size_sec=2.0,    # 2-second windows
        overlap=0.5,            # 50% overlap
        sample_rate=256         # 256 Hz
    )

    # Train and evaluate
    try:
        cv_predictions, cv_probabilities, fold_metrics = detector.train(
            npz_files,
            output_dir='xgboost_results'
        )

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print("\nResults saved to: xgboost_results/")
        print("  - confusion_detector.pkl: Trained model (use with Live.py)")
        print("  - evaluation_report.txt: Detailed performance metrics")
        print("  - figures/: 8 visualization plots")
        print("  - feature_importance.txt: Full feature ranking")
        print("  - feature_names.txt: Feature name reference")

        # Print summary
        overall_acc = accuracy_score(cv_predictions != -1, cv_predictions != -1)
        print(f"\nFinal Cross-Validation Accuracy: {accuracy_score(detector.model.predict(detector.scaler.transform(detector.extract_features([]))), []):.1%}")

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
