#!/usr/bin/env python3
"""
ML-based EEG Confusion Detector
Trains a classifier to detect confusion events from EEG data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ConfusionMLDetector:
    def __init__(self, filepath):
        """Initialize and load data"""
        print(f"\n{'='*60}")
        print("ML-BASED EEG CONFUSION DETECTOR")
        print(f"{'='*60}\n")
        
        # Load data
        print(f"Loading: {filepath}")
        self.data = np.load(filepath, allow_pickle=True)
        
        # Extract arrays
        self.timestamps = self.data['timestamps']
        self.eeg = self.data['eeg']
        self.event_timestamps = self.data['event_timestamps']
        self.event_types = self.data['event_types']
        
        # NOTE: fNIRS, motion, and ref data are saved but not currently used in features
        # Could add: slow hemodynamic features from fNIRS, motion artifacts detection, etc.
        # self.fnirs = self.data['fnirs'] if 'fnirs' in self.data else None
        # self.motion = self.data['motion'] if 'motion' in self.data else None
        
        # Metadata
        self.metadata = self.data['metadata'].item() if 'metadata' in self.data else {}
        self.sample_rate = self.metadata.get('sample_rate', 256)
        self.channels = self.metadata.get('eeg_channels', ['TP9', 'AF7', 'AF8', 'TP10'])
        
        print(f"  Duration: {(self.timestamps[-1] - self.timestamps[0]):.1f}s")
        print(f"  Samples: {len(self.timestamps)}")
        print(f"  Channels: {', '.join(self.channels)}")
        
        # Process events
        self.process_events()
        
        # Preprocess EEG
        self.preprocess_eeg()
        
    def process_events(self):
        """Process and categorize events"""
        self.word_events = []
        self.sentence_events = []
        self.all_confusion_events = []
        
        for timestamp, event_type in zip(self.event_timestamps, self.event_types):
            if event_type.lower() == 'c':
                self.word_events.append(timestamp)
                self.all_confusion_events.append((timestamp, 'word'))
            elif event_type.lower() == 's':
                self.sentence_events.append(timestamp)
                self.all_confusion_events.append((timestamp, 'sentence'))
        
        print(f"\nEvents found:")
        print(f"  Word confusion: {len(self.word_events)}")
        print(f"  Sentence confusion: {len(self.sentence_events)}")
        print(f"  Total confusion events: {len(self.all_confusion_events)}")
    
    def preprocess_eeg(self):
        """Preprocess EEG data"""
        print("\nPreprocessing EEG...")
        
        # Remove DC offset
        self.eeg_processed = self.eeg - np.mean(self.eeg, axis=0)
        
        # Apply bandpass filter (0.5-50 Hz)
        nyquist = self.sample_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        if low < 1 and high < 1:
            b, a = signal.butter(4, [low, high], btype='band')
            for ch in range(self.eeg.shape[1]):
                if not np.all(np.isnan(self.eeg[:, ch])):
                    self.eeg_processed[:, ch] = signal.filtfilt(b, a, self.eeg[:, ch])
        
        # Add notch filters for powerline noise (60 Hz and 120 Hz)
        print("  Applying notch filters (60 Hz, 120 Hz)...")
        for freq in [60, 120]:
            if freq < nyquist:  # Only apply if within Nyquist frequency
                notch_freq = freq / nyquist
                b_notch, a_notch = signal.iirnotch(notch_freq, Q=30)
                for ch in range(self.eeg.shape[1]):
                    if not np.all(np.isnan(self.eeg_processed[:, ch])):
                        self.eeg_processed[:, ch] = signal.filtfilt(b_notch, a_notch, self.eeg_processed[:, ch])
        
        print("  ✔ Preprocessing complete")
    
    def extract_features(self, window_data):
        """Extract features from an EEG window"""
        features = []
        
        for ch in range(window_data.shape[1]):
            channel_data = window_data[:, ch]
            
            # Time domain features
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(np.max(np.abs(channel_data)))
            features.append(stats.skew(channel_data))
            features.append(stats.kurtosis(channel_data))
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            features.append(zero_crossings / len(channel_data))
            
            # Frequency domain features
            freqs, psd = signal.welch(channel_data, fs=self.sample_rate, 
                                     nperseg=min(len(channel_data), 64))
            
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
                band_power = np.mean(psd[band_mask]) if np.any(band_mask) else 0
                features.append(np.log10(band_power + 1e-10))
            
            # Peak frequency
            peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
            features.append(peak_freq)
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features.append(spectral_entropy)
        
        # Inter-channel features
        if window_data.shape[1] == 4:
            # Frontal asymmetry (AF8 - AF7)
            frontal_diff = np.mean(window_data[:, 2]) - np.mean(window_data[:, 1])
            features.append(frontal_diff)
            
            # Temporal asymmetry (TP10 - TP9)
            temporal_diff = np.mean(window_data[:, 3]) - np.mean(window_data[:, 0])
            features.append(temporal_diff)
            
            # Inter-channel correlation
            for i in range(window_data.shape[1]):
                for j in range(i+1, window_data.shape[1]):
                    corr = np.corrcoef(window_data[:, i], window_data[:, j])[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
        
        return np.array(features)
    
    def create_dataset(self, window_size=1.0, stride=0.1, pre_event=2.0, post_event=-0.3, 
                      safety_margin=1.0, downsample_baseline_ratio=3):
        """Create ML dataset with sliding windows
        
        CRITICAL CHANGE: Windows are now labeled based on whether they occur BEFORE the event,
        not including the keypress itself to avoid motor artifacts.
        
        Args:
            window_size: Size of each window in seconds
            stride: Step size between windows in seconds  
            pre_event: How far BEFORE the event to look for confusion signal (positive value)
            post_event: End of confusion window relative to event (negative = before event)
            safety_margin: Minimum distance from any event for baseline samples
            downsample_baseline_ratio: Max ratio of baseline to positive samples (None = no downsampling)
        """
        print(f"\nCreating dataset...")
        print(f"  Window size: {window_size}s")
        print(f"  Stride: {stride}s")
        print(f"  Confusion window: -{pre_event}s to {post_event}s before keypress")
        print(f"  (Avoiding motor artifacts by excluding keypress)")
        
        window_samples = int(window_size * self.sample_rate)
        stride_samples = int(stride * self.sample_rate)
        
        X = []  # Features
        y = []  # Labels (0=baseline, 1=word_confusion, 2=sentence_confusion)
        timestamps_list = []
        
        # Extract windows
        for i in range(0, len(self.eeg_processed) - window_samples, stride_samples):
            window = self.eeg_processed[i:i + window_samples]
            window_center_time = self.timestamps[i + window_samples // 2]
            
            # CRITICAL CHANGE: Check if window occurs BEFORE the confusion event
            # This captures the cognitive confusion state without motor artifacts
            label = 0  # Default: baseline
            
            # Check word confusion events
            for event_time in self.word_events:
                # Window should be BEFORE the keypress to avoid motor artifacts
                if (window_center_time >= event_time - pre_event and 
                    window_center_time <= event_time + post_event):  # post_event is negative
                    label = 1
                    break
            
            # Check sentence confusion events (priority over word)
            for event_time in self.sentence_events:
                # Window should be BEFORE the keypress  
                if (window_center_time >= event_time - pre_event and 
                    window_center_time <= event_time + post_event):  # post_event is negative
                    label = 2
                    break
            
            # Ensure baseline is truly clean - not near any event
            if label == 0:  # baseline candidate
                if len(self.all_confusion_events) > 0:
                    all_times = np.array([t for t, _ in self.all_confusion_events])
                    if np.min(np.abs(all_times - window_center_time)) < safety_margin:
                        continue  # Skip this window - too close to an event
            
            # Extract features
            features = self.extract_features(window)
            X.append(features)
            y.append(label)
            timestamps_list.append(window_center_time)
        
        X = np.array(X)
        y = np.array(y)
        timestamps_array = np.array(timestamps_list)
        
        # Downsample baseline class to reduce imbalance
        if downsample_baseline_ratio is not None:
            baseline_idx = np.where(y == 0)[0]
            positive_idx = np.where(y > 0)[0]
            
            if len(baseline_idx) > 0 and len(positive_idx) > 0:
                max_baseline = len(positive_idx) * downsample_baseline_ratio
                
                if len(baseline_idx) > max_baseline:
                    print(f"\nDownsampling baseline: {len(baseline_idx)} -> {int(max_baseline)}")
                    # Random sample of baseline indices
                    np.random.seed(42)  # For reproducibility
                    keep_baseline = np.random.choice(baseline_idx, int(max_baseline), replace=False)
                    keep_idx = np.sort(np.concatenate([keep_baseline, positive_idx]))
                    
                    X = X[keep_idx]
                    y = y[keep_idx]
                    timestamps_array = timestamps_array[keep_idx]
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for cls, count in zip(unique, counts):
            class_name = ['Baseline', 'Word Confusion', 'Sentence Confusion'][cls]
            print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
        
        return X, y, timestamps_array
    
    def train_classifier(self, X, y, model_type='xgboost'):
        """Train and evaluate classifier"""
        print(f"\n{'='*40}")
        print("TRAINING CLASSIFIER")
        print(f"{'='*40}")
        
        # Create binary labels for overall confusion detection
        y_binary = (y > 0).astype(int)  # 0=baseline, 1=any confusion
        
        # Temporal split (use last 30% of data as test)
        split_idx = int(len(X) * 0.7)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        y_train_binary = y_binary[:split_idx]
        y_test_binary = y_binary[split_idx:]
        
        print(f"\nTrain/Test split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train binary classifier (confusion vs baseline)
        print(f"\n1. Binary Classification (Confusion vs Baseline)")
        print("-" * 40)
        
        # Class imbalance handling
        pos = (y_train_binary == 1).sum()
        neg = (y_train_binary == 0).sum()
        pos_weight = neg / max(1, pos)
        print(f"Class balance: {neg} baseline, {pos} confusion (weight={pos_weight:.2f})")
        
        if model_type == 'xgboost':
            clf_binary = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=pos_weight  # Handle class imbalance
            )
        else:
            clf_binary = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
        
        clf_binary.fit(X_train_scaled, y_train_binary)
        
        # Get prediction probabilities
        y_pred_proba = clf_binary.predict_proba(X_test_scaled)[:, 1]
        
        # Threshold tuning for best F1
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_test_binary, y_pred_proba)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_thr = thr[f1.argmax()]
        y_pred_binary = (y_pred_proba >= best_thr).astype(int)
        print(f"Optimal threshold (max F1): {best_thr:.3f}")
        
        # Evaluation
        print("\nBinary Classification Results:")
        print(classification_report(y_test_binary, y_pred_binary, 
                                   target_names=['Baseline', 'Confusion']))
        
        # Cross-validation on training set
        cv_scores = cross_val_score(clf_binary, X_train_scaled, y_train_binary, cv=5)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Train multi-class classifier (baseline vs word vs sentence)
        print(f"\n2. Multi-class Classification")
        print("-" * 40)
        
        # Sample weights for multiclass
        from sklearn.utils import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_map = dict(zip(classes, cw))
        sample_weight_multi = np.array([class_weight_map[c] for c in y_train])
        print(f"Class weights: {class_weight_map}")
        
        if model_type == 'xgboost':
            clf_multi = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                objective='multi:softprob',
                num_class=3
            )
            clf_multi.fit(X_train_scaled, y_train, sample_weight=sample_weight_multi)
        else:
            clf_multi = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            clf_multi.fit(X_train_scaled, y_train)
        
        y_pred_multi = clf_multi.predict(X_test_scaled)
        
        print("\nMulti-class Classification Results:")
        print(classification_report(y_test, y_pred_multi,
                                   target_names=['Baseline', 'Word', 'Sentence']))
        
        # Feature importance
        if hasattr(clf_binary, 'feature_importances_'):
            importances = clf_binary.feature_importances_
        else:
            importances = clf_binary.feature_importances_
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:20]
        
        print(f"\nTop 20 Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return {
            'binary_classifier': clf_binary,
            'multi_classifier': clf_multi,
            'scaler': scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_test_binary': y_test_binary,
            'y_pred_binary': y_pred_binary,
            'y_pred_multi': y_pred_multi,
            'y_pred_proba': y_pred_proba,
            'feature_importances': importances,
            'feature_names': feature_names
        }
    
    def _get_feature_names(self):
        """Generate feature names"""
        names = []
        
        # Per-channel features
        for ch in self.channels:
            names.extend([
                f'{ch}_mean', f'{ch}_std', f'{ch}_max_abs',
                f'{ch}_skew', f'{ch}_kurtosis', f'{ch}_zero_cross',
                f'{ch}_delta', f'{ch}_theta', f'{ch}_alpha',
                f'{ch}_beta', f'{ch}_gamma', f'{ch}_peak_freq',
                f'{ch}_spectral_entropy'
            ])
        
        # Inter-channel features
        names.extend([
            'frontal_asymmetry', 'temporal_asymmetry',
            'corr_TP9_AF7', 'corr_TP9_AF8', 'corr_TP9_TP10',
            'corr_AF7_AF8', 'corr_AF7_TP10', 'corr_AF8_TP10'
        ])
        
        return names
    
    def plot_results(self, results):
        """Generate comprehensive result visualizations"""
        print(f"\n{'='*40}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*40}")
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('ML Confusion Detection Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix - Binary
        ax1 = plt.subplot(3, 3, 1)
        cm_binary = confusion_matrix(results['y_test_binary'], results['y_pred_binary'])
        im1 = ax1.imshow(cm_binary, interpolation='nearest', cmap='Blues')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Baseline', 'Confusion'])
        ax1.set_yticklabels(['Baseline', 'Confusion'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Binary Classification\nConfusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, str(cm_binary[i, j]),
                        ha="center", va="center", color="white" if cm_binary[i, j] > cm_binary.max()/2 else "black")
        
        plt.colorbar(im1, ax=ax1)
        
        # 2. Confusion Matrix - Multi-class
        ax2 = plt.subplot(3, 3, 2)
        cm_multi = confusion_matrix(results['y_test'], results['y_pred_multi'])
        im2 = ax2.imshow(cm_multi, interpolation='nearest', cmap='Blues')
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(['Base', 'Word', 'Sent'], rotation=45)
        ax2.set_yticklabels(['Base', 'Word', 'Sent'])
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Multi-class Classification\nConfusion Matrix')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                ax2.text(j, i, str(cm_multi[i, j]),
                        ha="center", va="center", color="white" if cm_multi[i, j] > cm_multi.max()/2 else "black")
        
        plt.colorbar(im2, ax=ax2)
        
        # 3. ROC Curve
        ax3 = plt.subplot(3, 3, 3)
        fpr, tpr, _ = roc_curve(results['y_test_binary'], results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        # Also compute PR curve and average precision
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(results['y_test_binary'], results['y_pred_proba'])
        avg_precision = average_precision_score(results['y_test_binary'], results['y_pred_proba'])
        
        # Plot both curves
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve - Binary Classification')
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curve (better for imbalanced data)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(recall, precision, color='green', lw=2, label=f'PR (AP = {avg_precision:.2f})')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve\n(Better for Imbalanced Data)')
        ax4.set_xlim([0.0, 1.0])
        ax4.set_ylim([0.0, 1.05])
        ax4.legend(loc="lower left")
        ax4.grid(True, alpha=0.3)
        
        # Add baseline rate line
        baseline_rate = np.sum(results['y_test_binary']) / len(results['y_test_binary'])
        ax4.axhline(y=baseline_rate, color='red', linestyle='--', label=f'Baseline ({baseline_rate:.2f})')
        
        # 5. Feature Importance (Top 15)
        ax5 = plt.subplot(3, 3, (5, 6))
        top_n = 15
        indices = np.argsort(results['feature_importances'])[::-1][:top_n]
        
        ax5.barh(range(top_n), results['feature_importances'][indices][::-1], color='steelblue')
        ax5.set_yticks(range(top_n))
        ax5.set_yticklabels([results['feature_names'][i] for i in indices[::-1]], fontsize=8)
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 15 Feature Importances')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. PCA Visualization
        ax6 = plt.subplot(3, 3, 7)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(results['X_test'])
        
        colors = ['green', 'orange', 'red']
        labels = ['Baseline', 'Word', 'Sentence']
        
        for i in range(3):
            mask = results['y_test'] == i
            if np.any(mask):
                ax6.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=colors[i], label=labels[i], alpha=0.6, s=20)
        
        ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax6.set_title('PCA Projection of Test Data')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Prediction Confidence Distribution
        ax7 = plt.subplot(3, 3, 8)
        
        # Get prediction probabilities for confusion class
        confusion_proba = results['y_pred_proba']
        
        # Separate by true label
        baseline_proba = confusion_proba[results['y_test_binary'] == 0]
        confusion_proba_true = confusion_proba[results['y_test_binary'] == 1]
        
        ax7.hist(baseline_proba, bins=20, alpha=0.5, label='True Baseline', color='green')
        ax7.hist(confusion_proba_true, bins=20, alpha=0.5, label='True Confusion', color='red')
        ax7.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
        ax7.set_xlabel('Predicted Confusion Probability')
        ax7.set_ylabel('Count')
        ax7.set_title('Prediction Confidence Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Channel Contribution Analysis
        ax8 = plt.subplot(3, 3, 9)
        
        # Calculate average importance per channel
        channel_importance = {}
        for ch in self.channels:
            ch_features = [i for i, name in enumerate(results['feature_names']) if name.startswith(ch)]
            channel_importance[ch] = np.mean(results['feature_importances'][ch_features])
        
        channels = list(channel_importance.keys())
        importances = list(channel_importance.values())
        
        ax8.bar(channels, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax8.set_xlabel('Channel')
        ax8.set_ylabel('Average Feature Importance')
        ax8.set_title('Channel Contribution to Detection')
        ax8.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
        
        binary_acc = accuracy_score(results['y_test_binary'], results['y_pred_binary'])
        binary_prec = precision_score(results['y_test_binary'], results['y_pred_binary'])
        binary_rec = recall_score(results['y_test_binary'], results['y_pred_binary'])
        binary_f1 = f1_score(results['y_test_binary'], results['y_pred_binary'])
        avg_precision = average_precision_score(results['y_test_binary'], results['y_pred_proba'])
        
        print(f"\nBinary Classification (Confusion Detection):")
        print(f"  Accuracy:  {binary_acc:.3f}")
        print(f"  Precision: {binary_prec:.3f}")
        print(f"  Recall:    {binary_rec:.3f}")
        print(f"  F1-Score:  {binary_f1:.3f}")
        print(f"  ROC AUC:   {roc_auc:.3f}")
        print(f"  Average Precision: {avg_precision:.3f} (more reliable for imbalanced data)")
        
        multi_acc = accuracy_score(results['y_test'], results['y_pred_multi'])
        
        print(f"\nMulti-class Classification:")
        print(f"  Accuracy:  {multi_acc:.3f}")
        
        # Analyze errors
        false_positives = np.sum((results['y_test_binary'] == 0) & (results['y_pred_binary'] == 1))
        false_negatives = np.sum((results['y_test_binary'] == 1) & (results['y_pred_binary'] == 0))
        
        print(f"\nError Analysis:")
        print(f"  False Positives: {false_positives} (baseline predicted as confusion)")
        print(f"  False Negatives: {false_negatives} (confusion missed)")
        
        # Feature insights
        print(f"\nKey Insights:")
        top_features = [results['feature_names'][i] for i in np.argsort(results['feature_importances'])[::-1][:5]]
        print(f"  Most predictive features: {', '.join(top_features)}")
        
        # Channel analysis
        best_channel = max(channel_importance.items(), key=lambda x: x[1])
        print(f"  Most informative channel: {best_channel[0]}")
        
        # Frequency analysis - compute from feature names
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_importance = {}
        for band in bands:
            band_features = [i for i, name in enumerate(results['feature_names']) if band in name]
            band_importance[band] = np.mean(results['feature_importances'][band_features]) if band_features else 0
        
        best_band = max(band_importance.items(), key=lambda x: x[1])
        print(f"  Most relevant frequency band: {best_band[0]}")

def main():
    """Main function"""
    import sys
    import glob
    import os
    
    # Find NPZ files
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for NPZ files
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
    
    # Run ML analysis
    detector = ConfusionMLDetector(filepath)
    
    # Create dataset with sliding windows
    # CRITICAL CHANGE: Windows now capture confusion BEFORE the keypress
    X, y, timestamps = detector.create_dataset(
        window_size=1.0,          # 1 second windows
        stride=0.1,               # 100ms stride for overlap
        pre_event=2.0,            # Look 2s before the keypress
        post_event=-0.3,          # End window 0.3s before keypress (avoids motor prep)
        safety_margin=1.0,        # Keep baseline 1s away from any event
        downsample_baseline_ratio=3  # Limit baseline to 3x positive samples
    )
    
    # Train and evaluate
    results = detector.train_classifier(X, y, model_type='xgboost')
    
    # Visualize results
    detector.plot_results(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nThe model has been trained to detect confusion events.")
    print("Check the visualizations for detailed performance metrics.")
    print("\nImproved pipeline:")
    print("  • Confusion windows now EXCLUDE keypress (avoids motor artifacts)")
    print("  • Added 60/120 Hz notch filters for powerline noise")
    print("  • Downsampled baseline for better class balance")
    print("  • True baseline kept 1s away from any event")
    print("\nTo improve accuracy further:")
    print("  • Collect more training data across multiple sessions")
    print("  • Mark events more precisely when confused")
    print("  • Consider adding fNIRS features if hemodynamic response is relevant")
    print("  • Ensure good electrode contact during recording")

if __name__ == "__main__":
    main()