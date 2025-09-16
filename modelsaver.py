#!/usr/bin/env python3
"""
Save the trained confusion detection model for real-time use
Run this after training with the analysis script
"""

import pickle
import numpy as np
import sys

def save_model_for_realtime(training_results, output_path='confusion_model.pkl'):
    """Save the trained model and necessary components for real-time use"""
    
    # Package everything needed for real-time prediction
    model_package = {
        'classifier': training_results['classifier'],
        'scaler': training_results['scaler'],
        'feature_names': training_results['feature_names'],
        'feature_importances': training_results['feature_importances'],
        'metadata': {
            'device': 'Muse S Athena',
            'sample_rate': 256,
            'window_size': 2.0,  # seconds
            'eeg_channels': ['TP9', 'AF7', 'AF8', 'TP10'],
            'fnirs_channels': ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
        }
    }
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved to: {output_path}")
    print(f"Model type: {type(training_results['classifier']).__name__}")
    print(f"Features: {len(training_results['feature_names'])}")
    print(f"Classes: Baseline (0), Word Confusion (1), Sentence Confusion (2)")

# Example usage:
# After running the analysis script and getting the results dictionary:
# save_model_for_realtime(results, 'my_confusion_model.pkl')

if __name__ == "__main__":
    print("This script should be called from your analysis script after training.")
    print("Example:")
    print("  results = detector.train_word_model(X, y, words)")
    print("  save_model_for_realtime(results, 'confusion_model.pkl')")