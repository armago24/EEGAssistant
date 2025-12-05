#!/usr/bin/env python3
"""
RNN.py: Recurrent Neural Network for confusion detection from EEG/fNIRS data

This uses LSTM architecture which is appropriate for sequential time-series data
with temporal dependencies. Includes honest performance evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pickle

class EEGDataset(Dataset):
    """Dataset for EEG/fNIRS time-series windows with confusion labels"""

    def __init__(self, data_file, window_size=512, stride=64, confusion_window=1.0):
        """
        Args:
            data_file: Path to .npz file
            window_size: Number of samples per window (at 256 Hz, 512 = 2 seconds)
            stride: Samples to skip between windows (64 = 0.25 seconds overlap)
            confusion_window: Seconds around event to label as confused
        """
        print(f"\nLoading data from {data_file}...")
        data = np.load(data_file, allow_pickle=True)

        # Extract data
        self.eeg = data['eeg']  # (N, 4)
        self.fnirs = data['fnirs']  # (N, 8)
        self.motion = data['motion'] if 'motion' in data else np.zeros((len(self.eeg), 6))  # (N, 6)
        self.ref = data['ref'] if 'ref' in data else np.zeros((len(self.eeg), 2))  # (N, 2)
        self.timestamps = data['timestamps']

        # Combine all channels: 4 EEG + 8 fNIRS + 6 motion + 2 ref = 20 channels
        self.all_data = np.concatenate([self.eeg, self.fnirs, self.motion, self.ref], axis=1)

        print(f"Total samples: {len(self.all_data)}")
        print(f"Channels: {self.all_data.shape[1]} (4 EEG + 8 fNIRS + 6 motion + 2 ref)")

        # Create labels based on confusion events
        self.labels = self._create_labels(data, confusion_window)

        # Create windows
        self.windows, self.window_labels = self._create_windows(window_size, stride)

        print(f"Created {len(self.windows)} windows of size {window_size}")
        print(f"Label distribution: {np.bincount(self.window_labels)}")

        # Normalize data
        self._normalize()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Return (channels, time) for PyTorch Conv1d compatibility
        # But for LSTM we want (time, channels)
        window = torch.FloatTensor(self.windows[idx])  # (time, channels)
        label = torch.LongTensor([self.window_labels[idx]])[0]
        return window, label

    def _create_labels(self, data, confusion_window):
        """Create per-sample labels based on confusion events"""
        labels = np.zeros(len(self.timestamps), dtype=np.int64)

        if len(data['event_timestamps']) == 0:
            print("WARNING: No confusion events found in data")
            return labels

        # Mark samples near confusion events
        event_times = data['event_timestamps']
        event_types = data['event_types']

        confusion_count = 0
        for event_time, event_type in zip(event_times, event_types):
            # Find samples within confusion_window seconds of event
            time_diff = np.abs(self.timestamps - event_time)
            confused_samples = time_diff <= confusion_window

            # Label based on event type
            if 'word' in str(event_type):
                labels[confused_samples] = 1  # Word confusion
                confusion_count += np.sum(confused_samples)
            elif 'sentence' in str(event_type):
                labels[confused_samples] = 2  # Sentence confusion
                confusion_count += np.sum(confused_samples)

        baseline_count = np.sum(labels == 0)
        print(f"Labeled {confusion_count} confused samples, {baseline_count} baseline samples")
        print(f"Confusion ratio: {confusion_count / len(labels):.2%}")

        return labels

    def _create_windows(self, window_size, stride):
        """Create sliding windows from continuous data"""
        windows = []
        labels = []

        for i in range(0, len(self.all_data) - window_size, stride):
            window = self.all_data[i:i+window_size]

            # Label window by majority vote of samples
            window_labels = self.labels[i:i+window_size]
            label_counts = np.bincount(window_labels, minlength=3)

            # If more than 20% of window is confused, label as confused
            if label_counts[1] > window_size * 0.2:
                label = 1  # Word confusion
            elif label_counts[2] > window_size * 0.2:
                label = 2  # Sentence confusion
            else:
                label = 0  # Baseline

            windows.append(window)
            labels.append(label)

        return np.array(windows), np.array(labels)

    def _normalize(self):
        """Normalize windows per channel using StandardScaler"""
        n_windows, window_size, n_channels = self.windows.shape

        # Reshape to (n_windows * window_size, n_channels)
        reshaped = self.windows.reshape(-1, n_channels)

        # Fit scaler and transform
        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(reshaped)

        # Reshape back
        self.windows = normalized.reshape(n_windows, window_size, n_channels)

        print("Data normalized per channel")


class ConfusionLSTM(nn.Module):
    """
    LSTM-based confusion detection model with comprehensive regularization

    Regularization techniques:
    1. Dropout: Applied to LSTM layers and feedforward layers
    2. Layer Normalization: Stabilizes inputs and LSTM outputs
    3. Batch Normalization: Applied after first FC layer
    4. Weight Decay: L2 penalty on weights (applied in optimizer)
    5. Gradient Clipping: Prevents exploding gradients (applied in training)
    6. Label Smoothing: Prevents overconfident predictions (applied in loss)
    7. Early Stopping: Prevents overfitting to training set
    """

    def __init__(self, input_size=20, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(ConfusionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input layer normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Layer normalization after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)  # Less aggressive dropout for final layer

    def forward(self, x):
        # x shape: (batch, time, channels)

        # Input normalization
        x = self.input_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)

        # Layer normalization
        lstm_out = self.lstm_norm(lstm_out)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, time, 1)

        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Classification with regularization
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)

        return out


def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu', patience=10):
    """Train the model with early stopping and gradient clipping"""

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adam with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate F1 score (macro average for multi-class)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1_scores.append(val_f1)
        val_acc = accuracy_score(all_labels, all_preds)

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'confusion_rnn_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')

    plt.tight_layout()
    plt.savefig('rnn_training_history.png')
    print("\nTraining history saved to rnn_training_history.png")

    return best_val_f1


def evaluate_model(model, test_loader, device='cpu'):
    """Comprehensive honest evaluation of model performance"""

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "="*60)
    print("HONEST MODEL EVALUATION")
    print("="*60)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class metrics
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['Baseline', 'Word Confusion', 'Sentence Confusion'],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("Predicted:  Baseline  Word  Sentence")
    for i, row_label in enumerate(['Baseline', 'Word', 'Sentence']):
        print(f"{row_label:10s}  {cm[i,0]:5d}  {cm[i,1]:5d}  {cm[i,2]:5d}")

    # Class distribution in test set
    print("\nTest Set Class Distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = ['Baseline', 'Word Confusion', 'Sentence Confusion'][cls]
        print(f"  {class_name}: {count} ({count/len(all_labels)*100:.2f}%)")

    # Prediction distribution
    print("\nPrediction Distribution:")
    unique, counts = np.unique(all_preds, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = ['Baseline', 'Word Confusion', 'Sentence Confusion'][cls]
        print(f"  {class_name}: {count} ({count/len(all_preds)*100:.2f}%)")

    # Baseline comparison
    baseline_accuracy = np.max(counts) / len(all_labels)
    print(f"\nBaseline (always predict majority): {baseline_accuracy:.4f}")
    print(f"Model improvement over baseline: {(accuracy - baseline_accuracy)*100:.2f}%")

    # Average prediction confidence
    pred_confidences = np.max(all_probs, axis=1)
    print(f"\nAverage prediction confidence: {np.mean(pred_confidences):.4f}")
    print(f"Median prediction confidence: {np.median(pred_confidences):.4f}")

    # Confusion-specific analysis
    confusion_indices = all_labels > 0
    if np.sum(confusion_indices) > 0:
        confusion_accuracy = accuracy_score(all_labels[confusion_indices], all_preds[confusion_indices])
        print(f"\nAccuracy on confused samples only: {confusion_accuracy:.4f}")
    else:
        print("\nNo confused samples in test set")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Baseline', 'Word', 'Sentence'], rotation=45)
    plt.yticks(tick_marks, ['Baseline', 'Word', 'Sentence'])

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('rnn_confusion_matrix.png')
    print("\nConfusion matrix saved to rnn_confusion_matrix.png")

    print("\n" + "="*60)
    print("HONEST ASSESSMENT:")

    if accuracy < baseline_accuracy + 0.05:
        print("WARNING: Model is barely better than random guessing")
    elif accuracy < baseline_accuracy + 0.15:
        print("Model shows modest improvement over baseline")
    elif accuracy < 0.75:
        print("Model has moderate predictive power but significant room for improvement")
    else:
        print("Model shows strong predictive performance")

    if np.sum(confusion_indices) < 100:
        print("WARNING: Very few confusion samples in test set - results may not be reliable")

    print("="*60 + "\n")

    return accuracy, all_preds, all_labels, all_probs


def main():
    # Configuration
    WINDOW_SIZE = 512  # 2 seconds at 256 Hz
    STRIDE = 128  # 0.5 second stride
    CONFUSION_WINDOW = 1.0  # 1 second around event
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.4  # Increased for stronger regularization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nREGULARIZATION TECHNIQUES APPLIED:")
    print("  - Dropout: {:.1%} (recurrent + feedforward)".format(DROPOUT))
    print("  - Weight Decay (L2): 1e-4")
    print("  - Label Smoothing: 0.1")
    print("  - Gradient Clipping: max_norm=1.0")
    print("  - Layer Normalization: Input + LSTM output")
    print("  - Batch Normalization: FC layers")
    print("  - Early Stopping: patience=10 epochs")
    print("  - Learning Rate Scheduling: ReduceLROnPlateau")

    # List of data files to load
    data_files = [
        '/content/Post1.npz',
        '/content/Post2.npz',
        '/content/Post4.npz',
        '/content/Post5.npz',
        '/content/Post6.npz',
        '/content/Post7.npz'
    ]

    # Load and combine all data files
    all_windows = []
    all_labels = []
    all_scalers = []

    for data_file in data_files:
        dataset = EEGDataset(
            data_file,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            confusion_window=CONFUSION_WINDOW
        )
        all_windows.append(dataset.windows)
        all_labels.append(dataset.window_labels)
        all_scalers.append(dataset.scaler)

    # Combine datasets
    combined_windows = np.concatenate(all_windows, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    print(f"\nCombined dataset: {len(combined_windows)} windows")
    print(f"Label distribution: {np.bincount(combined_labels)}")

    # Create combined dataset
    class CombinedDataset(Dataset):
        def __init__(self, windows, labels):
            self.windows = windows
            self.labels = labels

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]

    full_dataset = CombinedDataset(combined_windows, combined_labels)

    # Split: 70% train, 15% validation, 15% test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nSplit: {train_size} train, {val_size} val, {test_size} test")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    input_size = combined_windows.shape[2]  # Number of channels
    model = ConfusionLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=3,
        dropout=DROPOUT
    ).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    print("\nStarting training...")
    best_val_f1 = train_model(model, train_loader, val_loader, NUM_EPOCHS, device)

    # Load best model
    model.load_state_dict(torch.load('confusion_rnn_best.pth'))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_preds, test_labels, test_probs = evaluate_model(model, test_loader, device)

    # Save model and metadata
    model_package = {
        'model_state_dict': model.state_dict(),
        'scalers': all_scalers,
        'config': {
            'input_size': input_size,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'confusion_window': CONFUSION_WINDOW
        },
        'performance': {
            'test_accuracy': test_accuracy,
            'best_val_f1': best_val_f1
        }
    }

    torch.save(model_package, 'confusion_rnn_model.pth')
    print("\nFull model package saved to confusion_rnn_model.pth")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()