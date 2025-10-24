#!/usr/bin/env python3
"""
Data Scrutinizer for EEG/fNIRS Confusion Detection Training Data
Analyzes NPZ files collected by eegtrainer.py
"""

import numpy as np
import os
from datetime import datetime
from pathlib import Path

def analyze_npz_file(filepath):
    """Analyze a single NPZ file and extract key metrics"""
    try:
        data = np.load(filepath, allow_pickle=True)

        # Extract arrays
        timestamps = data.get('timestamps', np.array([]))
        relative_timestamps = data.get('relative_timestamps', np.array([]))
        eeg = data.get('eeg', np.array([]))
        fnirs = data.get('fnirs', np.array([]))
        motion = data.get('motion', np.array([]))
        ref = data.get('ref', np.array([]))
        event_timestamps = data.get('event_timestamps', np.array([]))
        event_types = data.get('event_types', np.array([]))
        event_words = data.get('event_words', np.array([]))
        words = data.get('words', np.array([]))
        metadata = data.get('metadata', None)

        # Extract metadata if available
        if metadata is not None:
            metadata = metadata.item() if hasattr(metadata, 'item') else metadata

        # Calculate metrics
        metrics = {
            'filename': os.path.basename(filepath),
            'file_size_kb': os.path.getsize(filepath) / 1024,

            # Sample counts
            'total_samples': len(timestamps),
            'eeg_samples': len(eeg),
            'fnirs_samples': len(fnirs),
            'motion_samples': len(motion),
            'ref_samples': len(ref),

            # Duration
            'duration_seconds': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
            'duration_minutes': float(timestamps[-1] - timestamps[0]) / 60 if len(timestamps) > 1 else 0,

            # Sample rate
            'actual_sample_rate': len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0,
            'expected_sample_rate': metadata.get('sample_rate', 256) if metadata else 256,

            # Confusion events
            'total_events': len(event_timestamps),
            'word_confusion_events': np.sum(event_types == 'word_confusion') if len(event_types) > 0 else 0,
            'sentence_confusion_events': np.sum(event_types == 'sentence_confusion') if len(event_types) > 0 else 0,

            # Word tracking
            'total_word_samples': len(words),
            'unique_words_tracked': len(set(words[words != ''])) if len(words) > 0 else 0,
            'words_with_tracking': np.sum(words != '') if len(words) > 0 else 0,
            'word_tracking_percentage': (np.sum(words != '') / len(words) * 100) if len(words) > 0 else 0,

            # Data quality
            'eeg_nan_count': np.sum(np.isnan(eeg)) if len(eeg) > 0 else 0,
            'fnirs_nan_count': np.sum(np.isnan(fnirs)) if len(fnirs) > 0 else 0,
            'motion_nan_count': np.sum(np.isnan(motion)) if len(motion) > 0 else 0,
            'eeg_nan_percentage': (np.sum(np.isnan(eeg)) / eeg.size * 100) if eeg.size > 0 else 0,
            'fnirs_nan_percentage': (np.sum(np.isnan(fnirs)) / fnirs.size * 100) if fnirs.size > 0 else 0,

            # EEG channel statistics
            'eeg_mean': np.nanmean(eeg) if len(eeg) > 0 else 0,
            'eeg_std': np.nanstd(eeg) if len(eeg) > 0 else 0,
            'eeg_range': (np.nanmax(eeg) - np.nanmin(eeg)) if len(eeg) > 0 else 0,

            # Event density
            'events_per_minute': len(event_timestamps) / (timestamps[-1] - timestamps[0]) * 60 if len(timestamps) > 1 else 0,
            'samples_per_event': len(timestamps) / len(event_timestamps) if len(event_timestamps) > 0 else 0,

            # Metadata
            'text_passage_index': metadata.get('text_passage_index', 'N/A') if metadata else 'N/A',
            'start_time': datetime.fromtimestamp(metadata.get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S') if metadata else 'N/A',
        }

        # Calculate confusion event distribution over time
        if len(event_timestamps) > 0 and len(timestamps) > 1:
            event_relative_times = event_timestamps - timestamps[0]
            duration = timestamps[-1] - timestamps[0]

            # Split into thirds
            third = duration / 3
            metrics['events_first_third'] = np.sum(event_relative_times < third)
            metrics['events_middle_third'] = np.sum((event_relative_times >= third) & (event_relative_times < 2*third))
            metrics['events_last_third'] = np.sum(event_relative_times >= 2*third)
        else:
            metrics['events_first_third'] = 0
            metrics['events_middle_third'] = 0
            metrics['events_last_third'] = 0

        # Get confusion text examples
        if len(event_words) > 0:
            confusion_examples = [str(w)[:50] for w in event_words[:3]]  # First 3 examples
            metrics['confusion_examples'] = confusion_examples
        else:
            metrics['confusion_examples'] = []

        return metrics, True

    except Exception as e:
        return {'filename': os.path.basename(filepath), 'error': str(e)}, False

def write_report(all_metrics, output_file):
    """Write comprehensive report to file"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EEG TRAINER DATA COLLECTION SYSTEM - TECHNICAL SPECIFICATIONS\n")
        f.write("="*80 + "\n\n")

        f.write("SYSTEM OVERVIEW:\n")
        f.write("-" * 80 + "\n")
        f.write("Device: Muse S Athena (EEG/fNIRS headband)\n")
        f.write("Protocol: OSC (Open Sound Control) over UDP\n")
        f.write("Port: 8052\n")
        f.write("Sample Rate: 256 Hz (target)\n")
        f.write("Parser: Custom OSC parser (manual binary parsing)\n\n")

        f.write("DATA CHANNELS:\n")
        f.write("-" * 80 + "\n")
        f.write("EEG: 4 channels (TP9, AF7, AF8, TP10)\n")
        f.write("  - Temporal and frontal electrode positions\n")
        f.write("  - High-pass filtered (moving average baseline removal)\n")
        f.write("  - Units: microvolts (uV)\n\n")

        f.write("fNIRS/Optics: 8 channels total\n")
        f.write("  - 4 normalized channels (Ch1-4_norm)\n")
        f.write("  - 4 raw channels (Ch1-4_raw)\n")
        f.write("  - Measures blood oxygenation changes\n\n")

        f.write("Motion: 6-DOF inertial measurement\n")
        f.write("  - 3 accelerometer axes (acc_x, acc_y, acc_z)\n")
        f.write("  - 3 gyroscope axes (gyro_x, gyro_y, gyro_z)\n\n")

        f.write("Reference: 2 channels (DRL, REF)\n")
        f.write("  - Ground and reference electrode signals\n\n")

        f.write("LABELING METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("Mode Toggle: 'C' key switches between READING and LABELING modes\n")
        f.write("  - Reading mode: Data collection active, cursor tracking enabled\n")
        f.write("  - Labeling mode: Data collection PAUSED, confusion marking enabled\n\n")

        f.write("Confusion Marking:\n")
        f.write("  - LEFT-DRAG: Select multi-word phrases for 'word_confusion' label\n")
        f.write("  - RIGHT-CLICK: Mark entire sentence for 'sentence_confusion' label\n")
        f.write("  - Cursor tracking: Continuous word-under-cursor recording (50ms update)\n\n")

        f.write("Data Pausing:\n")
        f.write("  - When entering labeling mode, EEG/fNIRS recording PAUSES\n")
        f.write("  - This prevents contamination of baseline data with labeling artifacts\n")
        f.write("  - Recording resumes when returning to reading mode\n\n")

        f.write("NPZ FILE STRUCTURE:\n")
        f.write("-" * 80 + "\n")
        f.write("11 Arrays stored per session:\n")
        f.write("  1. timestamps: Absolute Unix timestamps (float64)\n")
        f.write("  2. relative_timestamps: Time relative to session start (float64)\n")
        f.write("  3. eeg: EEG data [N_samples x 4] (float32)\n")
        f.write("  4. fnirs: fNIRS data [N_samples x 8] (float32)\n")
        f.write("  5. motion: Motion data [N_samples x 6] (float32)\n")
        f.write("  6. ref: Reference data [N_samples x 2] (float32)\n")
        f.write("  7. event_timestamps: Confusion event times (float64)\n")
        f.write("  8. event_types: Event type strings ('word_confusion' or 'sentence_confusion')\n")
        f.write("  9. event_words: Text that was marked as confusing (object dtype)\n")
        f.write(" 10. words: Word at each sample timepoint (object dtype)\n")
        f.write(" 11. metadata: Dict with recording parameters (dict)\n\n")

        f.write("SYNCHRONIZATION:\n")
        f.write("-" * 80 + "\n")
        f.write("EEG packets drive the timeline (256 Hz master clock)\n")
        f.write("fNIRS and motion data interpolated to match EEG timestamps\n")
        f.write("Last-value interpolation fills gaps between non-EEG packets\n")
        f.write("NaN values indicate missing data for interpolated channels\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DATA SCRUTINY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files Analyzed: {len(all_metrics)}\n\n")

        # Summary statistics
        valid_metrics = [m for m in all_metrics if 'error' not in m]

        if valid_metrics:
            f.write("AGGREGATE STATISTICS:\n")
            f.write("-" * 80 + "\n")

            total_samples = sum(m['total_samples'] for m in valid_metrics)
            total_duration_min = sum(m['duration_minutes'] for m in valid_metrics)
            total_events = sum(m['total_events'] for m in valid_metrics)
            total_word_events = sum(m['word_confusion_events'] for m in valid_metrics)
            total_sentence_events = sum(m['sentence_confusion_events'] for m in valid_metrics)

            f.write(f"Total Recording Duration: {total_duration_min:.2f} minutes ({total_duration_min/60:.2f} hours)\n")
            f.write(f"Total Samples Collected: {total_samples:,}\n")
            f.write(f"Total Confusion Events: {total_events}\n")
            f.write(f"  - Word/Phrase Confusion: {total_word_events}\n")
            f.write(f"  - Sentence Confusion: {total_sentence_events}\n")
            f.write(f"Average Events per Session: {total_events/len(valid_metrics):.1f}\n")
            f.write(f"Average Session Duration: {total_duration_min/len(valid_metrics):.2f} minutes\n")
            f.write(f"Overall Event Density: {total_events/total_duration_min:.2f} events/minute\n\n")

            # Sample rate consistency
            avg_sample_rate = np.mean([m['actual_sample_rate'] for m in valid_metrics])
            std_sample_rate = np.std([m['actual_sample_rate'] for m in valid_metrics])
            f.write(f"Average Sample Rate: {avg_sample_rate:.2f} Hz (expected: 256 Hz)\n")
            f.write(f"Sample Rate Std Dev: {std_sample_rate:.2f} Hz\n")
            f.write(f"Sample Rate Consistency: {'GOOD' if std_sample_rate < 5 else 'POOR'}\n\n")

            # Data quality
            avg_eeg_nan = np.mean([m['eeg_nan_percentage'] for m in valid_metrics])
            avg_fnirs_nan = np.mean([m['fnirs_nan_percentage'] for m in valid_metrics])
            f.write(f"Average EEG NaN Percentage: {avg_eeg_nan:.2f}%\n")
            f.write(f"Average fNIRS NaN Percentage: {avg_fnirs_nan:.2f}%\n")
            f.write(f"Data Completeness: {'EXCELLENT' if avg_eeg_nan < 1 else 'ACCEPTABLE' if avg_eeg_nan < 5 else 'POOR'}\n\n")

            # Event distribution
            f.write("EVENT TEMPORAL DISTRIBUTION (within sessions):\n")
            first_third = sum(m['events_first_third'] for m in valid_metrics)
            middle_third = sum(m['events_middle_third'] for m in valid_metrics)
            last_third = sum(m['events_last_third'] for m in valid_metrics)
            f.write(f"  First Third: {first_third} events ({first_third/total_events*100:.1f}%)\n")
            f.write(f"  Middle Third: {middle_third} events ({middle_third/total_events*100:.1f}%)\n")
            f.write(f"  Last Third: {last_third} events ({last_third/total_events*100:.1f}%)\n\n")

        # Individual file reports
        f.write("\n" + "="*80 + "\n")
        f.write("INDIVIDUAL FILE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        for i, metrics in enumerate(all_metrics, 1):
            f.write(f"FILE {i}: {metrics['filename']}\n")
            f.write("-" * 80 + "\n")

            if 'error' in metrics:
                f.write(f"ERROR: {metrics['error']}\n\n")
                continue

            f.write(f"File Size: {metrics['file_size_kb']:.2f} KB\n")
            f.write(f"Recording Start: {metrics['start_time']}\n")
            f.write(f"Duration: {metrics['duration_minutes']:.2f} minutes ({metrics['duration_seconds']:.1f} seconds)\n")
            f.write(f"Text Passage: {metrics['text_passage_index']}\n\n")

            f.write("Sample Counts:\n")
            f.write(f"  Total Samples: {metrics['total_samples']:,}\n")
            f.write(f"  EEG Samples: {metrics['eeg_samples']:,}\n")
            f.write(f"  fNIRS Samples: {metrics['fnirs_samples']:,}\n")
            f.write(f"  Motion Samples: {metrics['motion_samples']:,}\n")
            f.write(f"  Reference Samples: {metrics['ref_samples']:,}\n\n")

            f.write("Sample Rate:\n")
            f.write(f"  Expected: {metrics['expected_sample_rate']:.2f} Hz\n")
            f.write(f"  Actual: {metrics['actual_sample_rate']:.2f} Hz\n")
            f.write(f"  Deviation: {abs(metrics['actual_sample_rate'] - metrics['expected_sample_rate']):.2f} Hz\n\n")

            f.write("Confusion Events:\n")
            f.write(f"  Total: {metrics['total_events']}\n")
            f.write(f"  Word/Phrase: {metrics['word_confusion_events']}\n")
            f.write(f"  Sentence: {metrics['sentence_confusion_events']}\n")
            f.write(f"  Events per minute: {metrics['events_per_minute']:.2f}\n")
            f.write(f"  Samples per event: {metrics['samples_per_event']:.1f}\n\n")

            f.write("Event Distribution:\n")
            f.write(f"  First third: {metrics['events_first_third']}\n")
            f.write(f"  Middle third: {metrics['events_middle_third']}\n")
            f.write(f"  Last third: {metrics['events_last_third']}\n\n")

            f.write("Word Tracking:\n")
            f.write(f"  Unique words tracked: {metrics['unique_words_tracked']}\n")
            f.write(f"  Samples with word tracking: {metrics['words_with_tracking']:,}\n")
            f.write(f"  Tracking coverage: {metrics['word_tracking_percentage']:.1f}%\n\n")

            f.write("Data Quality:\n")
            f.write(f"  EEG NaN count: {metrics['eeg_nan_count']} ({metrics['eeg_nan_percentage']:.2f}%)\n")
            f.write(f"  fNIRS NaN count: {metrics['fnirs_nan_count']} ({metrics['fnirs_nan_percentage']:.2f}%)\n")
            f.write(f"  Motion NaN count: {metrics['motion_nan_count']}\n\n")

            f.write("EEG Signal Statistics:\n")
            f.write(f"  Mean: {metrics['eeg_mean']:.2f} uV\n")
            f.write(f"  Std Dev: {metrics['eeg_std']:.2f} uV\n")
            f.write(f"  Range: {metrics['eeg_range']:.2f} uV\n\n")

            if metrics['confusion_examples']:
                f.write("Sample Confusion Events:\n")
                for j, example in enumerate(metrics['confusion_examples'], 1):
                    f.write(f"  {j}. \"{example}\"\n")
                f.write("\n")

            f.write("\n")

        # Assessment
        f.write("="*80 + "\n")
        f.write("ASSESSMENT FOR MACHINE LEARNING\n")
        f.write("="*80 + "\n\n")

        if valid_metrics:
            f.write("DATA SUFFICIENCY:\n")
            f.write("-" * 80 + "\n")

            # Check if enough data
            if total_samples < 50000:
                f.write("WARNING: Low total sample count. Recommend collecting more data.\n")
            elif total_samples < 150000:
                f.write("ACCEPTABLE: Moderate sample count. More data would improve model.\n")
            else:
                f.write("GOOD: Sufficient sample count for training.\n")

            # Check event balance
            event_ratio = total_word_events / total_sentence_events if total_sentence_events > 0 else 0
            f.write(f"\nEvent Balance Ratio (word:sentence): {event_ratio:.2f}:1\n")

            if total_events < 50:
                f.write("WARNING: Very few confusion events. Model may struggle to learn patterns.\n")
            elif total_events < 150:
                f.write("ACCEPTABLE: Moderate number of events. Consider collecting more.\n")
            else:
                f.write("GOOD: Sufficient confusion events for training.\n")

            # Class imbalance
            baseline_samples = total_samples - (total_events * 256 * 2)  # Approximate baseline
            confusion_samples = total_events * 256 * 2  # Approximate confusion
            imbalance_ratio = baseline_samples / confusion_samples if confusion_samples > 0 else 0
            f.write(f"\nEstimated Class Imbalance: {imbalance_ratio:.1f}:1 (baseline:confusion)\n")

            if imbalance_ratio > 100:
                f.write("WARNING: Severe class imbalance. Recommend using class weights or SMOTE.\n")
            elif imbalance_ratio > 50:
                f.write("CAUTION: High class imbalance. Use appropriate balancing techniques.\n")
            else:
                f.write("ACCEPTABLE: Class imbalance manageable with standard techniques.\n")

            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")

            if total_duration_min < 30:
                f.write("- Collect more recording sessions (target: 1+ hour total)\n")

            if total_events < 100:
                f.write("- Mark more confusion events per session\n")

            if avg_eeg_nan > 5:
                f.write("- Check device connection quality (high NaN percentage)\n")

            if std_sample_rate > 10:
                f.write("- Investigate sample rate inconsistencies\n")

            if last_third < first_third * 0.5:
                f.write("- Events concentrated early in sessions - may indicate fatigue/attention drop\n")

            f.write("\n")

def main():
    # File patterns to search
    tdata_dir = Path("/Users/AMago24/Documents/EEGAssistant/fixed_tdata")

    # Find all matching files
    file_patterns = [
        "Post1.npz", "Post2.npz", "Post3.npz", "Post4.npz",
        "Post5.npz", "Post6.npz", "Post7.npz"
    ]

    print("EEG Trainer Data Scrutinizer")
    print("="*80)
    print()

    all_metrics = []

    for pattern in file_patterns:
        filepath = tdata_dir / pattern

        if filepath.exists():
            print(f"Analyzing {filepath.name}...")
            metrics, success = analyze_npz_file(filepath)
            all_metrics.append(metrics)

            if success:
                print(f"  Duration: {metrics['duration_minutes']:.2f} min, "
                      f"Samples: {metrics['total_samples']:,}, "
                      f"Events: {metrics['total_events']}")
            else:
                print(f"  ERROR: {metrics.get('error', 'Unknown error')}")
        else:
            print(f"File not found: {filepath}")

    if all_metrics:
        print()
        print("Writing report to scrut_eegtrainer.txt...")
        output_path = "/Users/AMago24/Documents/EEGAssistant/scrut_eegtrainer.txt"
        write_report(all_metrics, output_path)
        print(f"Report written to: {output_path}")
        print()
        print("Summary:")
        valid = [m for m in all_metrics if 'error' not in m]
        print(f"  Files analyzed: {len(all_metrics)}")
        print(f"  Valid files: {len(valid)}")
        print(f"  Total samples: {sum(m['total_samples'] for m in valid):,}")
        print(f"  Total events: {sum(m['total_events'] for m in valid)}")
        print(f"  Total duration: {sum(m['duration_minutes'] for m in valid):.2f} minutes")
    else:
        print("No files found to analyze.")

if __name__ == "__main__":
    main()
