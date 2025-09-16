# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EEG/fNIRS-based confusion detection system that analyzes brain signals to identify when users are confused while reading. The system uses a Muse S Athena device connected via Bluetooth to an iPhone, which streams data via OSC (Open Sound Control) to Python applications.

## Core Architecture

### Data Flow
1. **Muse S Athena** (EEG/fNIRS device) → **iPhone** (Bluetooth) → **Python Code** (OSC over network)
2. **Real-time Processing**: OSC packets → Feature extraction → ML classification → UI visualization
3. **Training Pipeline**: Recorded data → Feature engineering → Model training → Saved models

### Key Components

**Data Collection & Processing:**
- `athena.py` - Standalone EEG/fNIRS visualizer with real-time plotting
- `eegTrainer.py` - Training data collector with text reading interface and click-based confusion labeling
- `packetChecker.py` - Low-level OSC packet analysis and debugging
- `viewer.py` - Basic data visualization tools

**Machine Learning Pipeline:**
- `Analysis.py` - Word-level confusion detection with comprehensive feature extraction (EEG bands, fNIRS hemodynamics, connectivity, word characteristics)
- `system.py` - Simplified live confusion detection system with real-time prediction
- `combined.py` - Real-time confusion detector combining data collection with ML inference
- `modelsaver.py` - Model persistence utilities

**Utilities:**
- `NoBS.py` - Streamlined data processing utilities
- `qualitysystem.py` - Data quality assessment and validation

### Data Storage Format

All training data is stored as NPZ files containing:
- `timestamps` - Sample timestamps
- `eeg` - 4-channel EEG data (TP9, AF7, AF8, TP10)
- `fnirs` - 8-channel fNIRS data (4 normalized + 4 raw)
- `motion` - 6-DOF accelerometer/gyroscope data
- `event_timestamps` - Confusion click timestamps
- `event_types` - Type of confusion ('word_confusion', 'sentence_confusion')
- `event_words` - Words that were clicked as confusing
- `words` - Continuous word timeline during reading
- `metadata` - Recording parameters and settings

## Common Development Commands

### Running the System

**Start real-time confusion detection:**
```bash
python system.py [training_data.npz]
```

**Train a new model:**
```bash
python Analysis.py [confusion_data.npz]
```

**Collect training data:**
```bash
python eegTrainer.py
```

**View live EEG/fNIRS data:**
```bash
python athena.py
```

**Real-time detection with visualization:**
```bash
python combined.py
```

### Data Analysis

**Check OSC packet format:**
```bash
python packetChecker.py
```

**Visualize recorded data:**
```bash
python viewer.py [data_file.npz]
```

**Quality assessment:**
```bash
python qualitysystem.py [data_file.npz]
```

## Key Technical Details

### Signal Processing
- **EEG**: 256 Hz sampling, 0.5-50 Hz bandpass filter, 60/120 Hz notch filters
- **fNIRS**: Lowpass filtered at 0.5 Hz for hemodynamic response capture
- **Feature Extraction**: Time domain (mean, std, skew, kurtosis), frequency domain (band powers), connectivity (phase sync, frontal asymmetry)

### OSC Protocol
- **Port**: 8052 (default)
- **Manual Parsing**: Custom OSC parser to avoid python-osc library issues
- **Message Types**: /eeg, /fnirs, /motion, /ref

### Machine Learning
- **Models**: XGBoost for classification, Random Forest for regression
- **Features**: 60+ features including EEG band powers, fNIRS hemodynamics, inter-channel connectivity, word characteristics
- **Classes**: 0=baseline, 1=word_confusion, 2=sentence_confusion

### Dependencies
- Core: `numpy`, `scipy`, `matplotlib`, `tkinter`
- ML: `scikit-learn`, `xgboost`
- Signal: `scipy.signal` for filtering and spectral analysis
- UI: `matplotlib` with TkAgg backend for real-time plotting

## File Naming Conventions

- `*.npz` - Training/recorded data files
- `mo*.pkl` - Saved trained models
- `*confusion*.npz` - Data files with click-based confusion labels
- `Teleprompter*.npz` - Earlier training datasets

## Development Notes

### Testing
- Set `test_mode = True` in system classes for development without EEG hardware
- Use mouse hover over text for confusion prediction testing
- Training texts include technical passages designed to induce confusion

### Real-time Performance
- 256 Hz sample rate requires efficient feature extraction
- Ring buffers (deque) manage memory for continuous data streams
- Threading separates data collection from processing/visualization

### Model Deployment
- Models are saved with scalers and metadata for consistent deployment
- Real-time prediction requires 2-second sliding windows
- Feature extraction must match training pipeline exactly

## Hardware Setup
- Muse S Athena device required for data collection
- iPhone app streams OSC data over WiFi
- Python applications receive on port 8052
- No additional dependencies for offline analysis of recorded data

## Important Note:

Never use unicode characters or emojis.
I like simple, effective, and clean solutions. The bare minimum to make the system work as I intend. Don't add or change things that aren't necessary. Do things the right way - don't just change the tests and output, change the core problems. Think deeply. Be rational.