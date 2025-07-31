import socket
import struct
from collections import deque, defaultdict
import threading
import time

class EEGfNIRSParser:
    def __init__(self, port=8052, buffer_size=1000):
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.running = False
        self.data_format_detected = False
        
        # Data buffers
        self.eeg_channels = {}
        self.fnirs_channels = {}
        self.timestamps = deque(maxlen=buffer_size)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Debug mode
        self.debug = True
        
    def parse_osc_string(self, data, offset):
        """Parse a null-terminated, 4-byte aligned string from OSC data"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        # Advance to next 4-byte boundary
        offset = ((end + 4) // 4) * 4
        return string, offset
    
    def parse_osc_message(self, data):
        """Manually parse an OSC message"""
        try:
            offset = 0
            
            # Parse address pattern (e.g., "/Aryan/eeg")
            address, offset = self.parse_osc_string(data, offset)
            if not address:
                return None
            
            # Parse type tag string (e.g., ",ffff")
            type_tags, offset = self.parse_osc_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            # Remove the leading comma
            type_tags = type_tags[1:]
            
            # Parse arguments based on type tags
            args = []
            for tag in type_tags:
                if tag == 'f':  # 32-bit float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 'i':  # 32-bit int
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>i', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 's':  # string
                    string, offset = self.parse_osc_string(data, offset)
                    args.append(string)
                # Add more types as needed
            
            return {
                'address': address,
                'args': args
            }
            
        except Exception as e:
            if self.debug:
                print(f"OSC parse error: {e}")
            return None
    
    def start_receiving(self):
        """Start UDP receiver thread"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)  # Non-blocking with timeout
            
            self.running = True
            self.receiver_thread = threading.Thread(target=self._receive_loop)
            self.receiver_thread.start()
            
            print(f"Started listening on UDP port {self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to start receiver: {e}")
            return False
    
    def _receive_loop(self):
        """Main receiver loop"""
        packet_count = 0
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                packet_count += 1
                
                if self.debug and packet_count % 100 == 0:
                    print(f"Packet {packet_count}: {len(data)} bytes from {addr}")
                    # Print hex dump of first few bytes
                    hex_str = ' '.join(f'{b:02x}' for b in data[:32])
                    print(f"  Data: {hex_str}...")
                
                # Try to parse as OSC
                message = self.parse_osc_message(data)
                if message:
                    self._process_osc_message(message)
                    
                    # Mark that we've successfully detected the format
                    if not self.data_format_detected:
                        self.data_format_detected = True
                        print(f"OSC data format detected! Address: {message['address']}")
                        self._print_detected_channels()
                else:
                    # Try to parse as OSC bundle (starts with "#bundle\0")
                    if data.startswith(b'#bundle\x00'):
                        self._parse_osc_bundle(data)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.debug:
                    print(f"Receiver error: {e}")
    
    def _parse_osc_bundle(self, data):
        """Parse an OSC bundle containing multiple messages"""
        try:
            if not data.startswith(b'#bundle\x00'):
                return
            
            # Skip "#bundle\0" (8 bytes) and timestamp (8 bytes)
            offset = 16
            
            while offset < len(data):
                # Read message size (4 bytes)
                if offset + 4 > len(data):
                    break
                    
                msg_size = struct.unpack('>i', data[offset:offset+4])[0]
                offset += 4
                
                if offset + msg_size > len(data):
                    break
                
                # Parse the message
                msg_data = data[offset:offset+msg_size]
                message = self.parse_osc_message(msg_data)
                if message:
                    self._process_osc_message(message)
                
                offset += msg_size
                
        except Exception as e:
            if self.debug:
                print(f"Bundle parse error: {e}")
                
    def _process_osc_message(self, message):
        """Process a parsed OSC message"""
        address = message['address']
        args = message['args']
        timestamp = time.time()
        
        with self.lock:
            # Store timestamp
            self.timestamps.append(timestamp)
            
            # Parse based on address pattern
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                username = parts[0]  # e.g., "Aryan"
                data_type = parts[1]  # e.g., "eeg", "acc", etc.
                
                if self.debug and len(self.timestamps) % 100 == 1:
                    print(f"  OSC: {address} -> {args}")
                
                # Process EEG data (4 channels: TP9, AF7, AF8, TP10)
                if data_type == 'eeg' and len(args) == 4:
                    channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
                    for name, value in zip(channel_names, args):
                        if name not in self.eeg_channels:
                            self.eeg_channels[name] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[name].append(value)
                
                # Process accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    axes = ['acc_x', 'acc_y', 'acc_z']
                    for axis, value in zip(axes, args):
                        if axis not in self.eeg_channels:
                            self.eeg_channels[axis] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[axis].append(value)
                
                # Process gyroscope data
                elif data_type == 'gyro' and len(args) == 3:
                    axes = ['gyro_x', 'gyro_y', 'gyro_z']
                    for axis, value in zip(axes, args):
                        if axis not in self.eeg_channels:
                            self.eeg_channels[axis] = deque(maxlen=self.buffer_size)
                        self.eeg_channels[axis].append(value)
                
                # Process optical/fNIRS data
                elif data_type == 'optics':
                    for i, value in enumerate(args):
                        channel_name = f'optical_{i+1}'
                        if channel_name not in self.fnirs_channels:
                            self.fnirs_channels[channel_name] = deque(maxlen=self.buffer_size)
                        self.fnirs_channels[channel_name].append(value)
                
                # Process DRL/REF data
                elif data_type == 'drlref' and len(args) >= 2:
                    if 'DRL' not in self.eeg_channels:
                        self.eeg_channels['DRL'] = deque(maxlen=self.buffer_size)
                    if 'REF' not in self.eeg_channels:
                        self.eeg_channels['REF'] = deque(maxlen=self.buffer_size)
                    self.eeg_channels['DRL'].append(args[0])
                    self.eeg_channels['REF'].append(args[1])
                    
    def _print_detected_channels(self):
        """Print detected channels"""
        print("\nDetected channels:")
        print(f"EEG channels: {list(self.eeg_channels.keys())}")
        print(f"fNIRS channels: {list(self.fnirs_channels.keys())}")
        
    def get_latest_data(self):
        """Get the latest data from all channels"""
        with self.lock:
            eeg_data = {ch: list(data) for ch, data in self.eeg_channels.items()}
            fnirs_data = {ch: list(data) for ch, data in self.fnirs_channels.items()}
            timestamps = list(self.timestamps)
            
        return eeg_data, fnirs_data, timestamps
    
    def stop(self):
        """Stop the receiver"""
        self.running = False
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join()
        if self.socket:
            self.socket.close()
        print("Parser stopped")

# Test the parser standalone
if __name__ == "__main__":
    parser = EEGfNIRSParser(port=8052)
    parser.debug = True  # Enable debug output
    
    print("Starting EEG/fNIRS Manual OSC Parser...")
    print("Listening for OSC messages on port 8052")
    print("Debug mode enabled - will show packet details")
    
    if parser.start_receiving():
        try:
            # Let it run for a while
            sample_count = 0
            while True:
                time.sleep(1)
                if parser.data_format_detected:
                    eeg_data, fnirs_data, timestamps = parser.get_latest_data()
                    if timestamps:
                        sample_count = len(timestamps)
                        print(f"\n=== Total samples received: {sample_count} ===")
                        
                        # Print latest EEG values
                        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
                            if ch in eeg_data and eeg_data[ch]:
                                print(f"{ch}: {eeg_data[ch][-1]:.2f} µV")
                        
                        # Print latest optical values
                        for i in range(1, 9):
                            ch = f'optical_{i}'
                            if ch in fnirs_data and fnirs_data[ch]:
                                print(f"{ch}: {fnirs_data[ch][-1]:.2f}")
                else:
                    print(".", end="", flush=True)
                        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            parser.stop()
    else:
        print("Failed to start parser")