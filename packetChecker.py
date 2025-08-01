#!/usr/bin/env python3
"""
Muse S Athena Packet Analyzer
Listens to UDP packets and analyzes their structure to understand the data format
"""

import socket
import struct
import time
import numpy as np
from collections import defaultdict, Counter
import json

class MuseSAthenPacketAnalyzer:
    def __init__(self, port=8052):
        self.port = port
        self.socket = None
        self.running = False
        
        # Analysis storage
        self.packet_count = 0
        self.message_types = defaultdict(int)
        self.message_samples = defaultdict(list)
        self.type_tag_patterns = defaultdict(int)
        self.address_patterns = defaultdict(int)
        self.value_ranges = defaultdict(list)
        self.packet_sizes = []
        
        # Raw packet storage for analysis
        self.raw_packets = []
        self.max_raw_packets = 100  # Store first 100 for analysis
        
    def parse_osc_string(self, data, offset):
        """Parse null-terminated, 4-byte aligned string from OSC data"""
        try:
            end = data.find(b'\x00', offset)
            if end == -1:
                return None, offset
            
            string = data[offset:end].decode('ascii')
            offset = ((end + 4) // 4) * 4
            return string, offset
        except:
            return None, offset
    
    def parse_osc_message(self, data):
        """Parse an OSC message and return detailed info"""
        try:
            offset = 0
            original_data = data
            
            # Parse address
            address, offset = self.parse_osc_string(data, offset)
            if not address:
                return None
            
            # Add leading / if missing
            if not address.startswith('/'):
                address = '/' + address
            
            # Parse type tags
            type_tags, offset = self.parse_osc_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            type_tags = type_tags[1:]  # Remove comma
            
            # Parse arguments
            args = []
            raw_values = []  # Include NaN values for analysis
            for i, tag in enumerate(type_tags):
                if tag == 'f':  # float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    raw_values.append(value)
                    if not np.isnan(value):
                        args.append(value)
                    offset += 4
                elif tag == 'i':  # int
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>i', data[offset:offset+4])[0]
                    args.append(value)
                    raw_values.append(value)
                    offset += 4
                elif tag == 'd':  # double
                    if offset + 8 > len(data):
                        break
                    value = struct.unpack('>d', data[offset:offset+8])[0]
                    raw_values.append(value)
                    if not np.isnan(value):
                        args.append(value)
                    offset += 8
                elif tag == 's':  # string
                    string_val, offset = self.parse_osc_string(data, offset)
                    if string_val is not None:
                        args.append(string_val)
                        raw_values.append(string_val)
            
            return {
                'address': address,
                'type_tags': type_tags,
                'args': args,
                'raw_values': raw_values,
                'total_values': len(raw_values),
                'non_nan_values': len(args),
                'nan_count': len(raw_values) - len(args),
                'packet_size': len(original_data),
                'raw_data': original_data[:100]  # First 100 bytes for analysis
            }
            
        except Exception as e:
            return {'error': str(e), 'raw_data': data[:100]}
    
    def analyze_message(self, message):
        """Analyze a parsed message and update statistics"""
        if 'error' in message:
            print(f"Parse error: {message['error']}")
            return
        
        address = message['address']
        type_tags = message['type_tags']
        args = message['args']
        raw_values = message['raw_values']
        
        # Track message types
        self.message_types[address] += 1
        
        # Track type tag patterns
        self.type_tag_patterns[type_tags] += 1
        
        # Track address patterns
        parts = address.strip('/').split('/')
        if len(parts) >= 2:
            data_type = parts[1]
            self.address_patterns[data_type] += 1
        
        # Store sample messages (first few of each type)
        if len(self.message_samples[address]) < 5:
            sample = {
                'type_tags': type_tags,
                'args': args,
                'raw_values': raw_values,
                'total_values': message['total_values'],
                'nan_count': message['nan_count'],
                'timestamp': time.time()
            }
            self.message_samples[address].append(sample)
        
        # Track value ranges for numeric data
        for i, val in enumerate(args):
            if isinstance(val, (int, float)) and not np.isnan(val):
                key = f"{address}_arg_{i}"
                self.value_ranges[key].append(val)
        
        # Track packet sizes
        self.packet_sizes.append(message['packet_size'])
    
    def print_analysis(self):
        """Print comprehensive analysis of observed packets"""
        print("\n" + "="*80)
        print("MUSE S ATHENA PACKET ANALYSIS")
        print("="*80)
        
        print(f"\nTotal packets analyzed: {self.packet_count}")
        print(f"Unique message types: {len(self.message_types)}")
        
        # Packet size analysis
        if self.packet_sizes:
            print(f"\nPacket sizes:")
            print(f"  Min: {min(self.packet_sizes)} bytes")
            print(f"  Max: {max(self.packet_sizes)} bytes")
            print(f"  Mean: {np.mean(self.packet_sizes):.1f} bytes")
            
            # Show size distribution
            size_counts = Counter(self.packet_sizes)
            print(f"  Most common sizes:")
            for size, count in size_counts.most_common(5):
                print(f"    {size} bytes: {count} packets")
        
        # Message type frequency
        print(f"\nMessage types by frequency:")
        for address, count in sorted(self.message_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.packet_count) * 100
            print(f"  {address}: {count} packets ({percentage:.1f}%)")
        
        # Data type patterns
        print(f"\nData types detected:")
        for data_type, count in sorted(self.address_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {data_type}: {count} packets")
        
        # Type tag patterns
        print(f"\nType tag patterns:")
        for tags, count in sorted(self.type_tag_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{tags}': {count} packets")
        
        # Detailed message analysis
        print(f"\n" + "-"*60)
        print("DETAILED MESSAGE STRUCTURE ANALYSIS")
        print("-"*60)
        
        for address in sorted(self.message_types.keys()):
            print(f"\n{address}:")
            samples = self.message_samples[address]
            
            if samples:
                # Analyze structure consistency
                type_tags_set = set(sample['type_tags'] for sample in samples)
                value_counts = [sample['total_values'] for sample in samples]
                nan_counts = [sample['nan_count'] for sample in samples]
                
                print(f"  Samples analyzed: {len(samples)}")
                print(f"  Type tag patterns: {type_tags_set}")
                print(f"  Total values per packet: {set(value_counts)}")
                print(f"  NaN values per packet: {set(nan_counts)}")
                
                # Show first sample in detail
                first_sample = samples[0]
                print(f"  Example packet:")
                print(f"    Type tags: '{first_sample['type_tags']}'")
                print(f"    Total values: {first_sample['total_values']}")
                print(f"    Valid values: {len(first_sample['args'])}")
                print(f"    NaN count: {first_sample['nan_count']}")
                
                if first_sample['args']:
                    print(f"    Value sample: {first_sample['args'][:10]}{'...' if len(first_sample['args']) > 10 else ''}")
                
                if first_sample['raw_values'] != first_sample['args']:
                    print(f"    Raw values: {first_sample['raw_values'][:10]}{'...' if len(first_sample['raw_values']) > 10 else ''}")
        
        # Value range analysis
        print(f"\n" + "-"*60)
        print("VALUE RANGE ANALYSIS")
        print("-"*60)
        
        for key in sorted(self.value_ranges.keys()):
            values = self.value_ranges[key]
            if len(values) > 5:  # Only show ranges with sufficient data
                print(f"\n{key}:")
                print(f"  Count: {len(values)}")
                print(f"  Range: {min(values):.3f} to {max(values):.3f}")
                print(f"  Mean: {np.mean(values):.3f}")
                print(f"  Std: {np.std(values):.3f}")
    
    def save_analysis(self, filename=None):
        """Save analysis results to JSON file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"muse_s_analysis_{timestamp}.json"
        
        # Prepare data for JSON serialization
        analysis_data = {
            'metadata': {
                'timestamp': time.time(),
                'total_packets': self.packet_count,
                'analysis_duration': time.time(),
                'device': 'Muse S Athena'
            },
            'message_types': dict(self.message_types),
            'address_patterns': dict(self.address_patterns),
            'type_tag_patterns': dict(self.type_tag_patterns),
            'packet_sizes': {
                'all_sizes': self.packet_sizes,
                'min': min(self.packet_sizes) if self.packet_sizes else 0,
                'max': max(self.packet_sizes) if self.packet_sizes else 0,
                'mean': float(np.mean(self.packet_sizes)) if self.packet_sizes else 0
            },
            'message_samples': {}
        }
        
        # Convert message samples to serializable format
        for address, samples in self.message_samples.items():
            analysis_data['message_samples'][address] = []
            for sample in samples:
                serializable_sample = {
                    'type_tags': sample['type_tags'],
                    'args': [float(x) if isinstance(x, np.floating) else x for x in sample['args']],
                    'total_values': sample['total_values'],
                    'nan_count': sample['nan_count'],
                    'timestamp': sample['timestamp']
                }
                analysis_data['message_samples'][address].append(serializable_sample)
        
        # Add value ranges
        analysis_data['value_ranges'] = {}
        for key, values in self.value_ranges.items():
            if len(values) > 0:
                analysis_data['value_ranges'][key] = {
                    'count': len(values),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'sample_values': [float(x) for x in values[:20]]  # First 20 values
                }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\nAnalysis saved to: {filename}")
        return filename
    
    def start_analysis(self, duration=30):
        """Start packet analysis for specified duration"""
        print(f"Starting Muse S Athena packet analysis...")
        print(f"Listening on UDP port {self.port} for {duration} seconds")
        print("Make sure your Muse S Athena is streaming data!")
        print("\nPress Ctrl+C to stop early\n")
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(1.0)
            self.running = True
            
            start_time = time.time()
            last_print = 0
            
            while self.running and (time.time() - start_time) < duration:
                try:
                    data, addr = self.socket.recvfrom(4096)
                    self.packet_count += 1
                    
                    # Store raw packet for analysis (first few)
                    if len(self.raw_packets) < self.max_raw_packets:
                        self.raw_packets.append(data)
                    
                    # Parse and analyze message
                    message = self.parse_osc_message(data)
                    if message:
                        self.analyze_message(message)
                    
                    # Print progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_print >= 5:
                        elapsed = current_time - start_time
                        remaining = duration - elapsed
                        print(f"Progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining, "
                              f"{self.packet_count} packets received")
                        last_print = current_time
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error processing packet: {e}")
                    continue
            
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
        except Exception as e:
            print(f"Error starting analysis: {e}")
            return
        finally:
            if self.socket:
                self.socket.close()
            self.running = False
        
        # Print analysis results
        self.print_analysis()
        
        # Save results
        filename = self.save_analysis()
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total analysis time: {time.time() - start_time:.1f} seconds")
        print(f"Packets per second: {self.packet_count / (time.time() - start_time):.1f}")
        print(f"Results saved to: {filename}")
        
        return filename


def main():
    analyzer = MuseSAthenPacketAnalyzer(port=8052)
    
    print("Muse S Athena Packet Analyzer")
    print("=" * 40)
    print("This tool will analyze UDP packets from your Muse S Athena device")
    print("to understand the data structure for both EEG and fNIRS signals.")
    print()
    
    try:
        duration = input("Enter analysis duration in seconds (default 30): ").strip()
        if duration:
            duration = int(duration)
        else:
            duration = 30
    except ValueError:
        duration = 30
    
    print(f"\nStarting {duration} second analysis...")
    print("Please ensure your Muse S Athena is connected and streaming data to UDP port 8052")
    
    try:
        filename = analyzer.start_analysis(duration)
        
        print(f"\nNext steps:")
        print(f"1. Review the analysis output above")
        print(f"2. Check the saved JSON file: {filename}")
        print(f"3. Use this information to modify the visualizer for fNIRS support")
        print(f"\nLook for new message types that might contain fNIRS data!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()