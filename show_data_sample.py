#!/usr/bin/env python3
"""
Script to export sample rows of EEG/fNIRS data from NPZ file to a readable text file.
Perfect for teaching data structure and format.
"""

import numpy as np
import os
from datetime import datetime

# Global file handle for output
output_file = None

def write_line(text=""):
    """Write a line to both file and console."""
    print(text)  # Also print to console
    if output_file:
        output_file.write(text + "\n")

def print_table(data, columns, max_rows=10):
    """Print data in a nice table format."""
    # Print header
    header = "{:>3} ".format("Row")
    for col in columns:
        header += "{:>10} ".format(col[:10])  # Truncate long column names
    write_line(header)
    write_line("-" * len(header))
    
    # Print data rows
    for i in range(min(len(data), max_rows)):
        row = "{:>3} ".format(i)
        for j, val in enumerate(data[i]):
            if isinstance(val, float):
                if abs(val) < 10:
                    row += "{:>10.2f} ".format(val)
                elif abs(val) < 1000:
                    row += "{:>10.1f} ".format(val)
                else:
                    row += "{:>10.0f} ".format(val)
            else:
                row += "{:>10} ".format(str(val)[:10])
        write_line(row)

def show_data_sample(filepath="tdata/Post6.npz", num_rows=10):
    """
    Display sample rows of EEG/fNIRS data with proper column headers.
    
    Args:
        filepath (str): Path to NPZ file
        num_rows (int): Number of rows to display
    """
    global output_file
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(filepath).replace('.npz', '')
    output_filename = f"data_sample_{base_name}_{timestamp}.txt"
    
    # Open output file
    output_file = open(output_filename, 'w', encoding='utf-8')
    
    write_line("EEG/fNIRS Data Sample Viewer")
    write_line("=" * 80)
    
    # Load the data
    data = np.load(filepath, allow_pickle=True)
    metadata = data['metadata'].item()
    
    write_line(f"File: {filepath}")
    write_line(f"Device: {metadata['device']}")
    write_line(f"Duration: {metadata['duration']:.1f} seconds")
    write_line(f"Sample Rate: {metadata['sample_rate']} Hz")
    write_line(f"Total Samples: {metadata['total_samples']:,}")
    write_line()
    
    # Extract the main data arrays
    timestamps = data['relative_timestamps'][:num_rows]
    eeg_data = data['eeg'][:num_rows]
    fnirs_data = data['fnirs'][:num_rows]
    motion_data = data['motion'][:num_rows]
    
    # Get channel names from metadata
    eeg_channels = metadata['eeg_channels']
    fnirs_channels = metadata['fnirs_channels']
    motion_channels = metadata['motion_channels']
    
    write_line("=" * 80)
    write_line("SAMPLE DATA (First {} rows)".format(num_rows))
    write_line("=" * 80)
    
    # Create column headers
    columns = ['Time_Rel'] + eeg_channels + fnirs_channels + motion_channels
    
    # Combine all data
    combined_data = np.column_stack([
        timestamps,
        eeg_data,
        fnirs_data, 
        motion_data
    ])
    
    write_line("📊 COMBINED DATA TABLE:")
    print_table(combined_data, columns)
    
    write_line()
    write_line("=" * 80)
    write_line("DATA BREAKDOWN BY SENSOR TYPE")
    write_line("=" * 80)
    
    # Show EEG data separately
    write_line()
    write_line("🧠 EEG DATA (microvolts):")
    eeg_with_time = np.column_stack([timestamps, eeg_data])
    print_table(eeg_with_time, ['Time_Rel'] + eeg_channels)
    
    # Show fNIRS data separately  
    write_line()
    write_line("🔴 fNIRS DATA (optical density):")
    fnirs_with_time = np.column_stack([timestamps, fnirs_data])
    print_table(fnirs_with_time, ['Time_Rel'] + fnirs_channels)
    
    # Show motion data separately
    write_line()
    write_line("📱 MOTION DATA (accelerometer: m/s², gyroscope: deg/s):")
    motion_with_time = np.column_stack([timestamps, motion_data])
    print_table(motion_with_time, ['Time_Rel'] + motion_channels)
    
    write_line()
    write_line("=" * 80)
    write_line("COLUMN EXPLANATIONS")
    write_line("=" * 80)
    write_line("Time_Rel:    Relative timestamp (seconds from start)")
    write_line("TP9, AF7, AF8, TP10:  EEG electrode positions (10-20 system)")
    write_line("Ch1-4_norm:   fNIRS normalized optical density channels")  
    write_line("Ch1-4_raw:    fNIRS raw optical density channels")
    write_line("acc_x/y/z:    Accelerometer data (m/s²)")
    write_line("gyro_x/y/z:   Gyroscope data (degrees/second)")
    
    # Show data types and ranges
    write_line()
    write_line("=" * 80)
    write_line("DATA STATISTICS")
    write_line("=" * 80)
    
    write_line(f"EEG Range:    {eeg_data.min():.1f} to {eeg_data.max():.1f} μV")
    write_line(f"fNIRS Range:  {fnirs_data.min():.2f} to {fnirs_data.max():.2f}")
    write_line(f"Motion Range: {motion_data.min():.2f} to {motion_data.max():.2f}")
    
    # Show some events if they exist
    if len(data['event_timestamps']) > 0:
        write_line()
        write_line("=" * 80)
        write_line("SAMPLE EVENTS")
        write_line("=" * 80)
        event_times = data['event_timestamps'][:5] - data['timestamps'][0]  # Relative to start
        event_types = data['event_types'][:5]
        event_words = data['event_words'][:5]
        
        write_line("{:>3} {:>10} {:>15} {:>15}".format("Row", "Time_Rel", "Event_Type", "Word"))
        write_line("-" * 50)
        for i in range(len(event_times)):
            write_line("{:>3} {:>10.2f} {:>15} {:>15}".format(
                i, event_times[i], str(event_types[i]), str(event_words[i])[:15]
            ))
    
    # Close the output file
    if output_file:
        output_file.close()
        print(f"\n✅ Data exported to: {output_filename}")
        print(f"📁 You can now copy the contents from {output_filename}")
    
    return output_filename

if __name__ == "__main__":
    try:
        filename = show_data_sample()
        print(f"\n✅ Success! Data has been exported to: {filename}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if output_file:
            output_file.close()
