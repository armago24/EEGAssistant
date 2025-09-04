#!/usr/bin/env python3
"""
Visualizer for Muse S Athena with EEG and fNIRS (optics) data
Modified for ML training with confusion detection and text display
Saves data as NPZ files with event timestamps
"""

import socket
import struct
import threading
import time
import numpy as np
from collections import deque
from scipy import signal
import matplotlib
try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for better rendering
except:
    pass  # Fall back to default backend if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import json
import os
from datetime import datetime
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import font as tkfont
import signal as sig
import atexit
import sys

# Training text with deliberately confusing elements
TRAINING_TEXTS = [
"""People trust their sense of timing until a metronome exposes how elastic it really is. Walkers who think they stride a steady beat find their pace drifting when a mild slope, a shadow, or a stray conversation pulls attention sideways. The mind predicts each interval by blending recent experience with an internal baseline, a quiet form of Bayesian smoothing that feels like common sense. Yet when the environment supplies an extra cue, such as the click of a turn signal or the hush of a library, people sometimes become less accurate, not more, because the added cue competes with the one they were already using. The result is not chaos but a small, lawful bias, a kind of cognitive parallax that can be measured. Even the breath couples to these judgments: exhalation lengthens perceived time a little, as if respiration were a metronome’s hidden escapement. A rare gust of anxiety can do the opposite, tightening the seconds like a drumhead, a mild form of tachypsychia.""",

"""A physical pendulum seems simpler. Give it a push, watch it swing, and the period follows straightforward rules about length and gravity. But real pendulums live in air, pick up minute torques, and pass through magnetic fields that trace faint, invisible isogons. At tiny amplitudes the motion tends toward linearity; at larger arcs the approximation frays, and the period stretches. Counterintuitively, observing more closely can, in delicate setups, change the motion itself: light needed to measure the bob imparts momentum, and a warm sensor ever so slightly stirs the plenum it sits in. Engineers work around this with baffling, damping, and isolation, but the act of mitigation alters the system they intend to leave untouched. Add a second pendulum coupled by a spring and the rhythms sometimes synchronize spontaneously, while a slight, deliberate detuning keeps them independent longer than intuition predicts. Diminish friction too far, and the system becomes so sensitive that a whisper of turbulence inoculates it against neat prediction.""",

"""Language, which feels solid in the mouth, is equally slippery when watched too closely. People imagine that spelling mirrors speech, yet orthography lags and resists like a stubborn rheology. Pronunciations drift; letters fossilize; pragmatic conventions patch the gaps. The human ear parses syllables with quick heuristics, treating similar sounds as the same until a context demands sharper boundaries. When a community standardizes a new spelling, readers often hear a “new” pronunciation that was already there, just unnoticed, as though print retrofitted the sound. The reverse also occurs: a change in everyday speech makes an old spelling look suddenly perverse, though it had long served as a serviceable palimpsest of history. In classrooms, a well-meant focus on rules can briefly slow fluency, because rule rehearsal competes with the automaticity fluent reading relies on. Later, the same rules help resolve ambiguity more quickly, a delayed benefit that feels like a release of cognitive viscosity.""",

"""Walk a forest edge and the boundary looks crisp, but ecology hides in gradients. The soil shifts from loam to sand; light angles change; a damp ecotone emerges where moss keeps its own calendar. Fungal threads weave through roots in a mutable mycelium, trading minerals for sugars with a logic that seems like commerce but follows chemical gradients and stochastic proximity. Plants that appear isolated in plain view share resources through these filaments, and sometimes starve neighbors through allelopathy without ever “deciding” anything. Disturbance complicates the picture: a small fire or a fallen tree can raise diversity for a time, opening space without erasing memory, while an unusually gentle decade may reduce variety by allowing the most competitive species to overspread. A longer drought can reverse that pattern again, pruning dominants and letting tenacious colonists gain a foothold. Landscapes do not seek balance; they exhibit homeostasis only over intervals that later prove local, a moving equilibrium with lacunarity.""",

"""In mathematics, symmetry suggests constraint, and constraint suggests fewer possibilities. Yet patterns on a plane show the opposite more often than expected. Requiring a tiling to avoid periodic repetition narrows the allowable shapes, but the resulting mosaics can explode in variety, assembling order from rules that forbid too much order. A single shape can force nonrepeating coverage, an aperiodic monotile that looks simple until its neighborhood rules conspire to make long-range echoes without a fundamental repeat. Some tilings carry rotational symmetries of high dihedral order while refusing to translate neatly; their patches align like quasicrystals, showing sharp diffraction without lattice repetition. Add one gentle condition—say, a bound on angles at each vertex—and the counting problem can get harder, not easier, because the space of legal configurations becomes thin and brittle. Then, paradoxically, a stricter bound can make enumeration tractable again by slicing away troublesome, fractal-edged cases with a clean apothem.""",

"""Consider heat moving through a house on a still winter night. Insulation slows conduction, air sealing tamps down convection, and radiant barriers reflect long-wave energy. If a wall segment remains uninsulated—a thermal bridge—heat finds that shortcut first. Seal only the obvious gaps, and the remaining leaks draw stronger pressure differences that pull cold air through hidden chases. People assume that any added insulation monotonically reduces heat loss, yet partial upgrades can reorganize the flow field and temporarily increase convective cycling in an attic plenum, raising ice dam risk even as the average temperature rises. The remedy is not to avoid improvement but to match the interventions: balance ventilation, eliminate bridging, and keep vapor drive within safe limits so enthalpy gradients do not deposit moisture where wood fibers are most hygroscopic. In mild climates, the same strategies can overshoot, trapping heat in shoulder seasons and making natural stack effects do the unwanted work of a fan.""",

"""Rivers teach the same lesson with water instead of air. A channel carves its path by balancing gravity, bank strength, and sediment load, and the deepest line—the thalweg—meanders even in a uniform valley. Flow near the bed spirals in helical cells that move sand sideways as much as downstream, sculpting point bars and undercutting bends until an oxbow lake peels away. Straightening a reach makes it safer for a season, but the new speed amplifies scour downstream, trading one hazard for another. Add levees, and floods grow taller while visiting less often, storing risk for a future day when the crest exceeds the design crest by inches and the last ditch fails. Conversely, a small setback of banks and a roughened floodplain can slow high water enough to lower peaks without changing the valley’s average discharge at all. The sediment budget keeps its own ledger, where a single storm can erase a decade of gradual aggradation.""",

"""In markets, people chase efficiency with the same confidence engineers chase laminar flow, and they run into comparable surprises. Raising a price usually lowers demand, but goods with strong signaling value or poor substitutes can bend that curve, making elasticity look piecewise rather than smooth. Incentives rescue many problems, though not the ones that change how an activity is perceived. A tiny payment for a task that once felt meaningful can crowd out the intrinsic draw, shrinking participation even as the budget increases. Remove the payment later, and participation may not rebound, because the task now carries the odor of a chore. At the same time, modest, nonmonetary recognition can expand effort beyond what the recognition could “buy,” much like a catalyst that shifts a reaction pathway without appearing in the final products. Systems that appear to need precision sometimes prefer coarse rules that participants can predict, because predictable imperfections reduce strategic oscillation.""",

"""Music offers a cleaner test bed for intention and outcome. A player keeps tempo, adjusts dynamics, and shapes phrases, yet ensemble tightness often improves when nobody “tries” to lead. Microdelays created by room acoustics make individual entries appear tardy to some ears and early to others, and still the group converges. Counterintuitively, slightly higher reverberation can make timing feel crisper, because sustained sound stitches small gaps, turning discrete attacks into a perceived continuum. Meanwhile, instruments that share a tuning system collide in subtle ways: equal temperament equalizes semitones but spreads minute beating across every interval, while just intonation perfects some chords at the expense of others. Performers learn to shade pitches to minimize roughness, a live compromise that theory books relegate to footnotes. Spectators think the melody carries the day; performers know the inaudible floor of overtones, infrasound stage rumble, and psychoacoustic masking can sway the emotional arc more than an extra decibel.""",

"""On a calm beach, waves look predictable until you try to model them. Shallow water transforms long swells into breaking crests as dispersion slows the troughs and steepens the face, and a sandbar acts like a prism for wavelengths. Two trains of similar period combine to beat against each other, making sets that surfers count as if they had agency. Add wind fetched over miles, and capillary ripples ride along the larger gravity waves, changing the surface roughness that the wind then grips. Sometimes damping the wind locally with a breakwater increases the height of waves on the far side by changing interference patterns, producing a lee that is not actually calmer. Sediment moves in pulses that lag behind storms, so a beach can shrink after fair weather and rebuild in wild months, an inversion driven by the direction of peak energy. The shoreline is a memory device with poor foresight but excellent archival fidelity.""",

"""Attention flickers like a candle, yet people navigate crowded sidewalks without colliding most of the time. The brain blends motion vectors, obstacle predictions, and a soft rule that keeps right-of-way ambiguous enough for last-second negotiation. Give walkers explicit lanes and some will resist, paradoxically increasing near-misses as they assert priority that the old ambiguity dissolved. At intersections, eye contact can reduce safety when it conveys false assurance, especially where curb geometry invites late decisions. The trick is not to remove clarity but to place it where it changes behavior before commitment—far enough upstream that course corrections are cheap. This is why a small change in texture underfoot, a tactile rille that hints at an edge, can outperform another sign. People like to credit their conscious monitoring; most of the benefit comes from constraints that make dangerous moves feel effortful, an embodied form of friction that encourages smoother flow without explicit instruction.""",

"""At night, the sky looks fixed, but sight itself changes over minutes. Cones in the retina hand off to rods as light dims, shifting sensitivity toward wavelengths that daytime color names conceal. During full dark adaptation, scotopic vision sharpens detection of faint objects at the cost of detail and hue, and averted gaze helps because the most rod-rich region sits off center. Amateur observers learn that a dim red lamp, which seems like a compromise, can improve accuracy: it preserves rod sensitivity while giving enough light for charts, so the overall session yields more. Telescopes obey similar tradeoffs. A larger aperture gathers more light and resolution, but turbulent air smears the image into a seeing disk that makes magnification moot beyond a threshold. On rare, steady nights the limit drops back to optics and focus tolerance; on most nights the atmosphere sets a ceiling that training can approach but never exceed. The Moon’s umbra during an eclipse makes the lesson literal: the shadow moves with planetary geometry, not with the drama human eyes project onto it, while surface albedo changes make the Earth’s edge look crisp or ragged depending on cloud bands that observers cannot control.""",

"""Throughout these examples, a pattern returns. Systems invite intervention, but they also route around it, carrying inertia and feedback that reward modesty. More measurement can help, but the process may alter what is being measured in small, lawful ways. Rules clarify, and sometimes they constrain the exact capacities they aim to amplify. In many domains, the elegant path is not the straightest but the one that leaves enough slack for adjustment, the slack that keeps oscillations from growing. Even when outcomes look perverse, they often track hidden gradients—of pressure, information, energy, or incentive—that we can map with patience. When we trace those gradients carefully, the odd reversals begin to feel ordinary, like familiar riffs in a long composition. And if a conclusion seems to vanish when examined too directly, it may not be wrong; it may be built atop phenomena that need a softer gaze, the kind of attention that steadies rather than tightens.""",
]

class TeleprompterWindow:
    """Separate window for displaying text in large format"""
    def __init__(self, parent_visualizer):
        self.parent = parent_visualizer
        self.root = tk.Tk()
        self.root.title("📖 READING MATERIAL - Confusion Detection Training")
        
        # Make window large
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        
        # Track if window is active
        self.active = True
        
        # Header frame
        header_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        # Title label
        title_label = tk.Label(header_frame, 
                               text="📖 CONFUSION DETECTION TRAINING",
                               font=('Arial', 24, 'bold'),
                               fg='#FFD93D',
                               bg='#1a1a1a')
        title_label.pack(pady=10)
        
        # Instructions label
        instructions = tk.Label(header_frame,
                               text="Press C for word confusion | Press S for sentence confusion",
                               font=('Arial', 14),
                               fg='#4ECDC4',
                               bg='#1a1a1a')
        instructions.pack()
        
        # Main text frame
        text_frame = tk.Frame(self.root, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Text display with scrollbar
        self.text_display = tk.Text(text_frame,
                                    font=('Georgia', 28, 'normal'),
                                    bg='#0a0a0a',
                                    fg='white',
                                    wrap=tk.WORD,
                                    padx=40,
                                    pady=30,
                                    spacing1=10,  # Space before each line
                                    spacing2=8,   # Space between wrapped lines
                                    spacing3=10,  # Space after each line
                                    insertwidth=0,  # Hide cursor
                                    highlightthickness=0,
                                    borderwidth=0,
                                    relief=tk.FLAT)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Make text read-only
        self.text_display.config(state=tk.DISABLED)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=100)
        status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)
        
        # Status labels
        self.text_status = tk.Label(status_frame,
                                   text=f"Text: 1/{len(TRAINING_TEXTS)}",
                                   font=('Arial', 16),
                                   fg='#96CEB4',
                                   bg='#1a1a1a')
        self.text_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.recording_status = tk.Label(status_frame,
                                        text="⏸ NOT RECORDING",
                                        font=('Arial', 16, 'bold'),
                                        fg='#888888',
                                        bg='#1a1a1a')
        self.recording_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.event_status = tk.Label(status_frame,
                                    text="Events: Word=0, Sentence=0",
                                    font=('Arial', 16),
                                    fg='#E74C3C',
                                    bg='#1a1a1a')
        self.event_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Navigation hints
        nav_label = tk.Label(status_frame,
                           text="↑/↓: Scroll | ←/→: Change Text | Space: Start/Stop Recording",
                           font=('Arial', 12),
                           fg='#888888',
                           bg='#1a1a1a')
        nav_label.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Position tracking for smooth scrolling
        self.scroll_position = 0.0
        
        # Update the display
        self.update_display()
        
    def on_key_press(self, event):
        """Handle keyboard events in teleprompter window"""
        if event.char.lower() == 'c':
            self.parent.record_event('c')
            self.flash_event("WORD")
        elif event.char.lower() == 's':
            self.parent.record_event('s')
            self.flash_event("SENTENCE")
        elif event.char in ['1', '2', '3']:
            self.parent.record_event(event.char)
            self.flash_event(f"EVENT {event.char}")
        elif event.keysym == 'Up':
            self.scroll_text(-0.05)
        elif event.keysym == 'Down':
            self.scroll_text(0.05)
        elif event.keysym == 'Left':
            self.parent.previous_text()
            self.update_display()
        elif event.keysym == 'Right':
            self.parent.next_text()
            self.update_display()
        elif event.keysym == 'space':
            self.parent.toggle_recording()
            self.update_status()
        elif event.char.lower() == 'q':
            self.on_close()
        elif event.char == '+' or event.char == '=':
            # Increase font size
            current_font = self.text_display.cget('font')
            if isinstance(current_font, str):
                font_parts = current_font.split()
                current_size = int(font_parts[1]) if len(font_parts) > 1 else 28
            else:
                current_size = 28
            new_size = min(current_size + 2, 48)
            self.text_display.config(font=('Georgia', new_size, 'normal'))
        elif event.char == '-':
            # Decrease font size
            current_font = self.text_display.cget('font')
            if isinstance(current_font, str):
                font_parts = current_font.split()
                current_size = int(font_parts[1]) if len(font_parts) > 1 else 28
            else:
                current_size = 28
            new_size = max(current_size - 2, 16)
            self.text_display.config(font=('Georgia', new_size, 'normal'))
    
    def flash_event(self, event_type):
        """Flash the screen briefly to indicate event recorded"""
        original_bg = self.text_display.cget('bg')
        flash_color = '#2a2a2a' if event_type == "WORD" else '#1a2a2a'
        self.text_display.config(bg=flash_color)
        self.root.after(100, lambda: self.text_display.config(bg=original_bg))
    
    def scroll_text(self, amount):
        """Scroll the text display smoothly"""
        self.text_display.yview_scroll(int(amount * 10), "units")
    
    def update_display(self):
        """Update the text display with current text"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        
        # Get current text
        current_text = TRAINING_TEXTS[self.parent.current_text_index]
        
        # Insert text with some formatting
        self.text_display.insert('1.0', current_text)
        
        # Make read-only again
        self.text_display.config(state=tk.DISABLED)
        
        # Reset scroll position
        self.text_display.yview_moveto(0)
        
        # Update status
        self.text_status.config(text=f"Text: {self.parent.current_text_index + 1}/{len(TRAINING_TEXTS)}")
    
    def update_status(self):
        """Update recording and event status"""
        if self.parent.is_recording:
            elapsed = time.time() - self.parent.recording_start_time
            self.recording_status.config(
                text=f"⏺ RECORDING: {elapsed:.1f}s",
                fg='#ff4444'
            )
            
            # Count events
            word_count = sum(1 for _, event_type in self.parent.recorded_events 
                           if event_type.lower() == 'c')
            sentence_count = sum(1 for _, event_type in self.parent.recorded_events 
                              if event_type.lower() == 's')
            
            self.event_status.config(
                text=f"Events: Word={word_count}, Sentence={sentence_count}"
            )
        else:
            self.recording_status.config(
                text="⏸ NOT RECORDING",
                fg='#888888'
            )
    
    def on_close(self):
        """Handle window close"""
        self.active = False
        self.root.destroy()
    
    def update_loop(self):
        """Regular update loop for status"""
        if self.active:
            self.update_status()
            self.root.after(100, self.update_loop)

class MuseAthenaVisualizer:
    def __init__(self, port=8052, buffer_size=2000, window_duration=10):
        # Network settings
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = buffer_size
        self.window_duration = window_duration
        self.timestamps = deque(maxlen=buffer_size)
        
        # Channel data storage - 4 EEG channels for Athena
        self.eeg_channels = {
            'TP9': deque(maxlen=buffer_size),
            'AF7': deque(maxlen=buffer_size),
            'AF8': deque(maxlen=buffer_size),
            'TP10': deque(maxlen=buffer_size)
        }
        
        # fNIRS/Optics channels - 8 values
        self.fnirs_channels = {
            'Ch1_norm': deque(maxlen=buffer_size),
            'Ch2_norm': deque(maxlen=buffer_size),
            'Ch3_norm': deque(maxlen=buffer_size),
            'Ch4_norm': deque(maxlen=buffer_size),
            'Ch1_raw': deque(maxlen=buffer_size),
            'Ch2_raw': deque(maxlen=buffer_size),
            'Ch3_raw': deque(maxlen=buffer_size),
            'Ch4_raw': deque(maxlen=buffer_size)
        }
        
        self.motion_channels = {
            'acc_x': deque(maxlen=buffer_size),
            'acc_y': deque(maxlen=buffer_size),
            'acc_z': deque(maxlen=buffer_size),
            'gyro_x': deque(maxlen=buffer_size),
            'gyro_y': deque(maxlen=buffer_size),
            'gyro_z': deque(maxlen=buffer_size)
        }
        
        self.ref_channels = {
            'DRL': deque(maxlen=buffer_size),
            'REF': deque(maxlen=buffer_size)
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Visualization
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.spectral_lines = {}
        
        # Teleprompter window
        self.teleprompter = None
        
        # Text navigation for training
        self.current_text_index = 0
        
        # Color schemes
        self.eeg_colors = {
            'TP9': '#FF6B6B',
            'AF7': '#4ECDC4',
            'AF8': '#45B7D1',
            'TP10': '#96CEB4'
        }
        
        self.fnirs_colors = {
            'Ch1': '#E74C3C',
            'Ch2': '#3498DB',
            'Ch3': '#2ECC71',
            'Ch4': '#F39C12'
        }
        
        # Spectral analysis parameters
        self.sample_rate = 256  # Muse EEG sample rate
        self.spectral_window_size = 512  # FFT window size
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        self.max_freq = 70  # Maximum frequency to display
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Recording functionality - optimized for performance
        self.is_recording = False
        self.recording_start_time = None
        self.record_button = None
        
        # Data storage for ML - using regular lists for efficiency
        self.recorded_timestamps = []
        self.recorded_eeg = []
        self.recorded_fnirs = []
        self.recorded_motion = []
        self.recorded_ref = []
        self.recorded_events = []  # List of (timestamp, event_type) tuples
        
        # For tracking the last saved data
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Flag to track if we're in the process of shutting down
        self.shutting_down = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup_on_exit)
        sig.signal(sig.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nReceived interrupt signal. Saving data if recording...")
        self.cleanup_on_exit()
        sys.exit(0)
    
    def cleanup_on_exit(self):
        """Cleanup function called on exit"""
        if self.shutting_down:
            return
        self.shutting_down = True
        
        if self.is_recording and len(self.recorded_timestamps) > 0:
            print("\nRecording in progress. Auto-saving data...")
            # Auto-save synchronously since we're exiting
            self.save_recording_npz(auto_save=True)
        
        self.running = False
        if self.socket:
            self.socket.close()
    
    def parse_osc_string(self, data, offset):
        """Parse null-terminated, 4-byte aligned string from OSC data"""
        end = data.find(b'\x00', offset)
        if end == -1:
            return None, offset
        
        string = data[offset:end].decode('ascii')
        offset = ((end + 4) // 4) * 4
        return string, offset
    
    def parse_osc_message(self, data):
        """Parse an OSC message"""
        try:
            offset = 0
            
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
            
            type_tags = type_tags[1:]
            
            # Parse arguments
            args = []
            for tag in type_tags:
                if tag == 'f':  # float
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
                elif tag == 'i':  # int
                    if offset + 4 > len(data):
                        break
                    value = struct.unpack('>i', data[offset:offset+4])[0]
                    args.append(value)
                    offset += 4
            
            return {'address': address, 'args': args, 'type_tags': type_tags}
            
        except Exception as e:
            return None
    
    def process_osc_message(self, message):
        """Process incoming OSC message"""
        address = message['address']
        args = message['args']
        type_tags = message['type_tags']
        timestamp = time.time()
        
        with self.lock:
            # Parse address: /username/datatype
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                # EEG data - 4 floats for Athena
                if data_type == 'eeg' and len(args) == 4:
                    # Add timestamp
                    self.timestamps.append(timestamp)
                    
                    channels = ['TP9', 'AF7', 'AF8', 'TP10']
                    for ch, val in zip(channels, args):
                        self.eeg_channels[ch].append(val)
                    self.eeg_packet_count += 1
                    
                    # Store for recording
                    self.last_eeg_data = args
                    
                    # Record data if recording - simplified for performance
                    if self.is_recording:
                        # Only append essential data
                        self.recorded_timestamps.append(timestamp)
                        self.recorded_eeg.append(args)
                        
                        # Append last known values for other channels
                        self.recorded_fnirs.append(self.last_fnirs_data if self.last_fnirs_data else [np.nan] * 8)
                        self.recorded_motion.append(self.last_motion_data if self.last_motion_data else [np.nan] * 6)
                        self.recorded_ref.append(self.last_ref_data if self.last_ref_data else [np.nan] * 2)
                    
                    # Debug first few packets
                    if self.eeg_packet_count <= 5:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                # fNIRS/Optics data - 8 floats
                elif data_type == 'optics' and len(args) == 8:
                    norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
                    raw_channels = ['Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw']
                    
                    for i, (ch, val) in enumerate(zip(norm_channels + raw_channels, args)):
                        self.fnirs_channels[ch].append(val)
                    
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                    
                    # Debug first few packets
                    if self.fnirs_packet_count <= 5:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}, raw={args[4:]}")
                
                # Accelerometer data
                elif data_type == 'acc' and len(args) == 3:
                    channels = ['acc_x', 'acc_y', 'acc_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    # Update last motion data (first 3 values)
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[:3] = args
                
                # Gyroscope data
                elif data_type == 'gyro' and len(args) == 3:
                    channels = ['gyro_x', 'gyro_y', 'gyro_z']
                    for ch, val in zip(channels, args):
                        self.motion_channels[ch].append(val)
                    
                    # Update last motion data (last 3 values)
                    if self.last_motion_data is None:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[3:] = args
                
                # DRL/REF data
                elif data_type == 'drlref' and len(args) >= 2:
                    self.ref_channels['DRL'].append(args[0])
                    self.ref_channels['REF'].append(args[1])
                    self.last_ref_data = args[:2]
    
    def receiver_loop(self):
        """Main UDP receiver loop"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                self.packet_count += 1
                
                # Try to parse as OSC message
                message = self.parse_osc_message(data)
                if message:
                    self.process_osc_message(message)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def next_text(self):
        """Move to the next text passage"""
        self.current_text_index = (self.current_text_index + 1) % len(TRAINING_TEXTS)
        print(f"\n📖 Switched to text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.update_display()
    
    def previous_text(self):
        """Move to the previous text passage"""
        self.current_text_index = (self.current_text_index - 1) % len(TRAINING_TEXTS)
        print(f"\n📖 Switched to text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.update_display()
    
    def setup_visualization(self):
        """Setup matplotlib figure and axes"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid: 6 rows for different data types (removed text display)
        gs = GridSpec(6, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        # Main plots
        self.axes['eeg'] = self.fig.add_subplot(gs[0, 0])
        self.axes['spectral'] = self.fig.add_subplot(gs[1, 0])
        self.axes['fnirs'] = self.fig.add_subplot(gs[2, 0])
        self.axes['motion'] = self.fig.add_subplot(gs[3, 0])
        self.axes['gyro'] = self.fig.add_subplot(gs[4, 0])
        self.axes['ref'] = self.fig.add_subplot(gs[5, 0])
        
        # Info panel
        self.axes['info'] = self.fig.add_subplot(gs[:5, 1])
        
        # Configure axes
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name != 'info':
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        
        # Titles and labels
        self.axes['eeg'].set_title('EEG Channels (4 channels)', fontsize=14, color='#4ECDC4', pad=10)
        self.axes['eeg'].set_ylabel('Amplitude (µV)')
        
        self.axes['spectral'].set_title('Spectral Analysis - Power Spectral Density', fontsize=14, color='#FFD93D', pad=10)
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        
        self.axes['fnirs'].set_title('fNIRS/Optics - Functional Near-Infrared Spectroscopy', fontsize=14, color='#E74C3C', pad=10)
        self.axes['fnirs'].set_ylabel('Intensity')
        
        self.axes['motion'].set_title('Accelerometer', fontsize=14, color='#96CEB4', pad=10)
        self.axes['motion'].set_ylabel('Acceleration (g)')
        
        self.axes['gyro'].set_title('Gyroscope', fontsize=14, color='#9B59B6', pad=10)
        self.axes['gyro'].set_ylabel('Angular velocity (°/s)')
        
        self.axes['ref'].set_title('Reference Electrodes', fontsize=12, color='#95A5A6', pad=10)
        self.axes['ref'].set_ylabel('Voltage')
        self.axes['ref'].set_xlabel('Time (s)')
        
        # Info panel setup
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        self.axes['info'].set_title('Signal Statistics', fontsize=14, color='#FFD93D', pad=10)
        
        # Add record button
        ax_button = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.record_button = Button(ax_button, 'Begin Recording', 
                                   color='#2a2a2a', hovercolor='#3a3a3a')
        self.record_button.on_clicked(self.toggle_recording)
        
        # Initialize plot lines
        self._initialize_lines()
        
        plt.tight_layout()
    
    def toggle_recording(self, event=None):
        """Toggle recording state"""
        if not self.is_recording:
            # Start recording
            with self.lock:
                self.is_recording = True
                self.recording_start_time = time.time()
                self.recorded_timestamps = []
                self.recorded_eeg = []
                self.recorded_fnirs = []
                self.recorded_motion = []
                self.recorded_ref = []
                self.recorded_events = []
            
            if self.record_button:
                self.record_button.label.set_text('Stop Recording')
                self.record_button.color = '#ff4444'
                self.record_button.hovercolor = '#ff6666'
            
            # Update teleprompter status
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.update_status()
            
            print(f"\n{'='*50}")
            print(f"RECORDING STARTED at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            print("\n🎯 CONFUSION MARKERS:")
            print("  'c' = Word confusion")
            print("  's' = Sentence confusion")
            print("  '1', '2', '3' = Other markers")
            print("\n📖 TEXT NAVIGATION (in teleprompter window):")
            print("  ↑/↓ = Scroll text")
            print("  ←/→ = Previous/Next passage")
            print("\nData will be saved as .npz file when recording stops")
        else:
            # Stop recording
            self.is_recording = False
            
            # Copy data for saving (do this before updating UI)
            with self.lock:
                save_data = {
                    'timestamps': self.recorded_timestamps.copy(),
                    'eeg': self.recorded_eeg.copy(),
                    'fnirs': self.recorded_fnirs.copy(),
                    'motion': self.recorded_motion.copy(),
                    'ref': self.recorded_ref.copy(),
                    'events': self.recorded_events.copy(),
                    'start_time': self.recording_start_time,
                    'current_text_index': self.current_text_index
                }
                data_points = len(self.recorded_timestamps)
                events_count = len(self.recorded_events)
            
            # Update UI immediately
            if self.record_button:
                self.record_button.label.set_text('Saving...')
                self.record_button.color = '#888888'
                self.record_button.hovercolor = '#888888'
            
            duration = time.time() - self.recording_start_time
            print(f"\n{'='*50}")
            print(f"RECORDING STOPPED")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Data points: {data_points}")
            print(f"Events marked: {events_count}")
            print(f"Text passage: {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
            print(f"{'='*50}")
            print("\nPreparing to save...")
            
            # Save in a separate thread to avoid blocking
            save_thread = threading.Thread(
                target=self._save_recording_thread,
                args=(save_data,)
            )
            save_thread.daemon = True
            save_thread.start()
    
    def record_event(self, event_type):
        """Record an event with timestamp"""
        if self.is_recording:
            timestamp = time.time()
            relative_time = timestamp - self.recording_start_time
            self.recorded_events.append((timestamp, event_type))
            
            # Special messages for confusion events
            if event_type.lower() == 'c':
                print(f"📝 WORD confusion marked at {relative_time:.2f}s")
            elif event_type.lower() == 's':
                print(f"📄 SENTENCE confusion marked at {relative_time:.2f}s")
            else:
                print(f"🔷 Event '{event_type}' marked at {relative_time:.2f}s")
            
            # Update teleprompter if active
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.update_status()
    
    def _save_recording_thread(self, save_data):
        """Save recording in a separate thread to avoid blocking"""
        try:
            # Show save dialog in thread-safe way
            filename = self._get_save_filename()
            
            if filename:
                # Update button to show progress
                if self.record_button:
                    self.record_button.label.set_text('Converting...')
                    plt.draw()  # Force UI update
                
                # Convert lists to numpy arrays (this is the slow part)
                print("Converting data to numpy arrays...")
                timestamps = np.array(save_data['timestamps'])
                eeg_data = np.array(save_data['eeg']) if save_data['eeg'] else np.array([])
                fnirs_data = np.array(save_data['fnirs']) if save_data['fnirs'] else np.array([])
                motion_data = np.array(save_data['motion']) if save_data['motion'] else np.array([])
                ref_data = np.array(save_data['ref']) if save_data['ref'] else np.array([])
                
                # Convert events
                if save_data['events']:
                    event_timestamps = np.array([e[0] for e in save_data['events']])
                    event_types = np.array([e[1] for e in save_data['events']])
                else:
                    event_timestamps = np.array([])
                    event_types = np.array([])
                
                # Calculate relative timestamps
                relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
                
                # Create metadata
                metadata = {
                    'device': 'Muse S Athena',
                    'start_time': float(save_data['start_time']),
                    'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                    'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                    'sample_rate': self.sample_rate,
                    'total_samples': len(timestamps),
                    'total_events': len(save_data['events']),
                    'text_passage_index': save_data['current_text_index'],
                    'text_passage': TRAINING_TEXTS[save_data['current_text_index']],
                    'eeg_channels': ['TP9', 'AF7', 'AF8', 'TP10'],
                    'fnirs_channels': ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm', 
                                     'Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw'],
                    'motion_channels': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                    'ref_channels': ['DRL', 'REF']
                }
                
                # Update button
                if self.record_button:
                    self.record_button.label.set_text('Writing file...')
                    plt.draw()  # Force UI update
                
                print("Saving to file...")
                # Save as compressed numpy file
                np.savez_compressed(
                    filename,
                    timestamps=timestamps,
                    relative_timestamps=relative_timestamps,
                    eeg=eeg_data,
                    fnirs=fnirs_data,
                    motion=motion_data,
                    ref=ref_data,
                    event_timestamps=event_timestamps,
                    event_types=event_types,
                    metadata=metadata
                )
                
                print(f"\n{'='*50}")
                print(f"✅ DATA SAVED SUCCESSFULLY")
                print(f"File: {filename}")
                print(f"Size: {os.path.getsize(filename) / 1024:.1f} KB")
                print(f"\nContents:")
                print(f"  - {len(timestamps)} timestamped samples")
                print(f"  - EEG data: {eeg_data.shape if eeg_data.size > 0 else 'None'}")
                print(f"  - fNIRS data: {fnirs_data.shape if fnirs_data.size > 0 else 'None'}")
                print(f"  - Motion data: {motion_data.shape if motion_data.size > 0 else 'None'}")
                print(f"  - Reference data: {ref_data.shape if ref_data.size > 0 else 'None'}")
                print(f"  - {len(event_timestamps)} events marked")
                print(f"  - Text passage: {save_data['current_text_index'] + 1}/{len(TRAINING_TEXTS)}")
                
                if len(event_timestamps) > 0:
                    print(f"\nEvent Summary:")
                    unique_events, counts = np.unique(event_types, return_counts=True)
                    for event, count in zip(unique_events, counts):
                        if event.lower() == 'c':
                            print(f"  📝 Word confusion: {count} times")
                        elif event.lower() == 's':
                            print(f"  📄 Sentence confusion: {count} times")
                        else:
                            print(f"  🔷 Event '{event}': {count} times")
                
                print(f"\nTo load this data:")
                print(f"  data = np.load('{os.path.basename(filename)}')")
                print(f"  eeg = data['eeg']")
                print(f"  events = data['event_timestamps']")
                print(f"  metadata = data['metadata'].item()")
                print(f"{'='*50}\n")
            else:
                print("Save cancelled")
        
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Reset button state
            if self.record_button:
                self.record_button.label.set_text('Begin Recording')
                self.record_button.color = '#2a2a2a'
                self.record_button.hovercolor = '#3a3a3a'
                plt.draw()  # Force final UI update
    
    def _get_save_filename(self):
        """Get save filename using file dialog"""
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        root.focus_force()
        
        default_name = f"muse_athena_confusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = filedialog.asksaveasfilename(
            parent=root,
            initialdir=os.path.expanduser("~/Downloads"),
            initialfile=default_name,
            defaultextension=".npz",
            filetypes=[("NumPy Compressed", "*.npz"), ("All files", "*.*")]
        )
        
        root.destroy()
        return filename
    
    def save_recording_npz(self, auto_save=False):
        """Save the recorded data to NPZ file (for auto-save)"""
        if len(self.recorded_timestamps) == 0:
            print("No data to save")
            return
        
        if auto_save:
            # Prepare data for saving
            save_data = {
                'timestamps': self.recorded_timestamps,
                'eeg': self.recorded_eeg,
                'fnirs': self.recorded_fnirs,
                'motion': self.recorded_motion,
                'ref': self.recorded_ref,
                'events': self.recorded_events,
                'start_time': self.recording_start_time,
                'current_text_index': self.current_text_index
            }
            
            # Auto-save without dialog
            default_dir = os.path.expanduser("~/Downloads")
            os.makedirs(default_dir, exist_ok=True)
            filename = os.path.join(default_dir, 
                                   f"muse_athena_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
            
            print(f"Auto-saving to: {filename}")
            
            # Convert and save
            timestamps = np.array(save_data['timestamps'])
            eeg_data = np.array(save_data['eeg']) if save_data['eeg'] else np.array([])
            fnirs_data = np.array(save_data['fnirs']) if save_data['fnirs'] else np.array([])
            motion_data = np.array(save_data['motion']) if save_data['motion'] else np.array([])
            ref_data = np.array(save_data['ref']) if save_data['ref'] else np.array([])
            
            if save_data['events']:
                event_timestamps = np.array([e[0] for e in save_data['events']])
                event_types = np.array([e[1] for e in save_data['events']])
            else:
                event_timestamps = np.array([])
                event_types = np.array([])
            
            relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
            
            metadata = {
                'device': 'Muse S Athena',
                'start_time': float(save_data['start_time']),
                'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps),
                'total_events': len(save_data['events']),
                'text_passage_index': save_data['current_text_index'],
                'text_passage': TRAINING_TEXTS[save_data['current_text_index']]
            }
            
            np.savez_compressed(
                filename,
                timestamps=timestamps,
                relative_timestamps=relative_timestamps,
                eeg=eeg_data,
                fnirs=fnirs_data,
                motion=motion_data,
                ref=ref_data,
                event_timestamps=event_timestamps,
                event_types=event_types,
                metadata=metadata
            )
            
            print(f"Auto-save complete: {filename}")
    
    def _initialize_lines(self):
        """Initialize all plot lines"""
        # EEG lines - 4 channels
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95,
                                         linestyle='-', marker='',
                                         antialiased=True)
            self.lines[f'eeg_{ch}'] = line
        
        # Initialize spectral plot
        self.spectral_lines = {}
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color, 
                                              linewidth=1.5, alpha=0.9,
                                              antialiased=True)
            self.spectral_lines[ch] = line
        
        # Set up spectral axis
        self.axes['spectral'].set_xlim(0, self.max_freq)
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        
        # Add frequency band labels
        for band_name, (low, high) in self.freq_bands.items():
            self.axes['spectral'].axvline(x=low, color='white', linestyle=':', alpha=0.3, linewidth=0.5)
            mid_freq = (low + high) / 2
            if mid_freq < self.max_freq:
                self.axes['spectral'].text(mid_freq, 0.98, band_name, 
                                         fontsize=8, color='white', alpha=0.7,
                                         horizontalalignment='center',
                                         verticalalignment='top',
                                         transform=self.axes['spectral'].get_xaxis_transform())
        
        # fNIRS lines - normalized values only for main display
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color,
                                          linewidth=1.5, alpha=0.9,
                                          antialiased=True)
            self.lines[f'fnirs_{ch_norm}'] = line
        
        # Motion lines - accelerometer
        acc_colors = ['#3498DB', '#2ECC71', '#9B59B6']
        for i, ch in enumerate(['acc_x', 'acc_y', 'acc_z']):
            line, = self.axes['motion'].plot([], [], label=ch.replace('acc_', ''),
                                           color=acc_colors[i],
                                           linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Gyroscope lines
        gyro_colors = ['#F39C12', '#E67E22', '#D35400']
        for i, ch in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            line, = self.axes['gyro'].plot([], [], label=ch.replace('gyro_', ''),
                                         color=gyro_colors[i],
                                         linewidth=1.5, alpha=0.9)
            self.lines[f'motion_{ch}'] = line
        
        # Reference lines
        ref_colors = ['#95A5A6', '#7F8C8D']
        for i, ch in enumerate(['DRL', 'REF']):
            line, = self.axes['ref'].plot([], [], label=ch,
                                        color=ref_colors[i],
                                        linewidth=1.5, alpha=0.9)
            self.lines[f'ref_{ch}'] = line
        
        # Add legends
        self.axes['eeg'].legend(loc='upper left', fontsize=8, ncol=4, 
                               framealpha=0.7, bbox_to_anchor=(0.3, 1))
        self.axes['spectral'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.7)
        self.axes['fnirs'].legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.5)
        self.axes['motion'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['gyro'].legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.5)
        self.axes['ref'].legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.5)
    
    def compute_spectrum(self, data, sample_rate=256):
        """Compute power spectral density using Welch's method"""
        if len(data) < self.spectral_window_size:
            return None, None
        
        # Use Welch's method for more stable spectrum estimation
        frequencies, psd = signal.welch(
            data, 
            fs=sample_rate, 
            nperseg=min(len(data), self.spectral_window_size),
            noverlap=min(len(data)//2, self.spectral_window_size//2),
            scaling='density'
        )
        
        # Limit to 0-70 Hz
        freq_mask = frequencies <= self.max_freq
        frequencies = frequencies[freq_mask]
        psd = psd[freq_mask]
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        return frequencies, psd_db
    
    def update_plot(self, frame):
        """Update all plots with latest data"""
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Get the minimum length across all EEG channels to ensure alignment
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Calculate time axis based on actual timestamps
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                # Use actual time differences
                time_axis = timestamps - timestamps[-1]  # Make most recent = 0
                
                # Calculate display window based on time
                display_mask = time_axis >= -self.window_duration
                display_samples = np.sum(display_mask)
            else:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Update EEG
            eeg_values_for_scaling = []
            filtered_eeg_data = {}
            
            for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        # Get aligned data
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
                        # Apply high-pass filter to remove DC
                        if len(data_array) > 50:
                            window_size = min(50, len(data_array) // 4)
                            if window_size > 1:
                                moving_avg = np.convolve(data_array, 
                                                       np.ones(window_size)/window_size, 
                                                       mode='same')
                                filtered_data = data_array - moving_avg
                            else:
                                filtered_data = data_array - np.mean(data_array)
                        else:
                            filtered_data = data_array - np.mean(data_array)
                        
                        # Store for spectral analysis
                        filtered_eeg_data[ch_name] = filtered_data
                        
                        # Apply display mask
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            # Update spectral analysis
            if len(filtered_eeg_data) == 4:
                all_psd_values = []
                
                for ch_name in ['TP9', 'AF7', 'AF8', 'TP10']:
                    if ch_name in filtered_eeg_data and ch_name in self.spectral_lines:
                        spectral_window_samples = min(len(filtered_eeg_data[ch_name]), 
                                                    int(self.sample_rate * 4))
                        data_for_spectrum = filtered_eeg_data[ch_name][-spectral_window_samples:]
                        
                        frequencies, psd = self.compute_spectrum(data_for_spectrum, self.sample_rate)
                        
                        if frequencies is not None and psd is not None:
                            self.spectral_lines[ch_name].set_data(frequencies, psd)
                            all_psd_values.extend(psd)
                
                # Auto-scale y-axis
                if all_psd_values:
                    y_min = np.percentile(all_psd_values, 5) - 5
                    y_max = np.percentile(all_psd_values, 95) + 5
                    self.axes['spectral'].set_ylim(y_min, y_max)
            
            # Update fNIRS - show normalized values
            fnirs_norm_channels = ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm']
            for ch_name in fnirs_norm_channels:
                if ch_name in self.fnirs_channels and len(self.fnirs_channels[ch_name]) > 0:
                    line_key = f'fnirs_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.fnirs_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Motion (accelerometer)
            for ch_name in ['acc_x', 'acc_y', 'acc_z']:
                if ch_name in self.motion_channels and len(self.motion_channels[ch_name]) > 0:
                    line_key = f'motion_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.motion_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Gyroscope
            for ch_name in ['gyro_x', 'gyro_y', 'gyro_z']:
                if ch_name in self.motion_channels and len(self.motion_channels[ch_name]) > 0:
                    line_key = f'motion_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.motion_channels[ch_name]))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update Reference
            for ch_name, data in self.ref_channels.items():
                if len(data) > 0:
                    line_key = f'ref_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(data))
                        if len(data_array) >= len(display_mask):
                            aligned_data = data_array[-len(display_mask):]
                            self.lines[line_key].set_data(
                                time_axis[display_mask],
                                aligned_data[display_mask]
                            )
            
            # Update axes limits
            for ax in [self.axes['eeg'], self.axes['fnirs'], 
                      self.axes['motion'], self.axes['gyro'], self.axes['ref']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            # Special handling for EEG scaling
            if eeg_values_for_scaling:
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                y_range = 4 * eeg_std
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
                
                self.axes['eeg'].text(0.02, 0.98, 
                                    f'Scale: ±{y_range/2:.1f} µV', 
                                    transform=self.axes['eeg'].transAxes, 
                                    fontsize=9,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Update info panel
            self._update_info_panel()
        
        return list(self.lines.values()) + list(self.spectral_lines.values())
    
    def _update_info_panel(self):
        """Update statistics panel"""
        self.axes['info'].clear()
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        y_pos = 0.95
        self.axes['info'].text(0.1, y_pos, 'Signal Statistics', 
                             fontsize=12, weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
        # Recording status
        if self.is_recording:
            y_pos -= 0.06
            elapsed = time.time() - self.recording_start_time
            self.axes['info'].text(0.1, y_pos, f'⏺ REC: {elapsed:.1f}s', 
                                 fontsize=10, color='#ff4444', weight='bold',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'Samples: {len(self.recorded_timestamps)}', 
                                 fontsize=9, color='#ff6666',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            
            # Count confusion events
            word_confusion = sum(1 for _, event_type in self.recorded_events 
                                if event_type.lower() == 'c')
            sentence_confusion = sum(1 for _, event_type in self.recorded_events 
                                   if event_type.lower() == 's')
            other_count = len(self.recorded_events) - word_confusion - sentence_confusion
            
            self.axes['info'].text(0.1, y_pos, f'📝 Word: {word_confusion}', 
                                 fontsize=9, color='#ffaa44',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'📄 Sentence: {sentence_confusion}', 
                                 fontsize=9, color='#ff8844',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            if other_count > 0:
                self.axes['info'].text(0.1, y_pos, f'🔷 Other: {other_count}', 
                                     fontsize=9, color='#66aaff',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.04
        
        # Text info
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, f'Text: {self.current_text_index + 1}/{len(TRAINING_TEXTS)}', 
                             fontsize=9, color='#aaaaaa',
                             transform=self.axes['info'].transAxes)
        
        # EEG stats
        y_pos -= 0.08
        self.axes['info'].text(0.1, y_pos, 'EEG (4ch):', fontsize=10, 
                             weight='bold', color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if ch in self.eeg_channels and len(self.eeg_channels[ch]) > 0:
                data = np.array(list(self.eeg_channels[ch])[-100:])
                mean_val = np.mean(data)
                std_val = np.std(data)
                self.axes['info'].text(0.15, y_pos, f'{ch}:', fontsize=9,
                                     color=self.eeg_colors[ch],
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.4, y_pos, f'{mean_val:.1f}±{std_val:.1f}',
                                     fontsize=9, color='white',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.04
        
        # Frequency band power
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, 'Band Power:', fontsize=10,
                             weight='bold', color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Calculate average band powers
        if hasattr(self, 'spectral_lines'):
            band_powers = {band: [] for band in self.freq_bands.keys()}
            
            for ch_name, line in self.spectral_lines.items():
                xdata, ydata = line.get_data()
                if len(xdata) > 0 and len(ydata) > 0:
                    for band_name, (low, high) in self.freq_bands.items():
                        band_mask = (xdata >= low) & (xdata <= high)
                        if np.any(band_mask):
                            band_power = np.mean(ydata[band_mask])
                            band_powers[band_name].append(band_power)
            
            for band_name in self.freq_bands.keys():
                if band_powers[band_name]:
                    avg_power = np.mean(band_powers[band_name])
                    self.axes['info'].text(0.15, y_pos, f'{band_name}:', fontsize=9,
                                         color='white',
                                         transform=self.axes['info'].transAxes)
                    self.axes['info'].text(0.4, y_pos, f'{avg_power:.1f} dB',
                                         fontsize=9, color='white',
                                         transform=self.axes['info'].transAxes)
                    y_pos -= 0.04
        
        # fNIRS stats
        y_pos -= 0.04
        self.axes['info'].text(0.1, y_pos, 'fNIRS:', fontsize=10,
                             weight='bold', color='#E74C3C',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        # Show both normalized and raw values
        for i, ch_base in enumerate(['Ch1', 'Ch2', 'Ch3', 'Ch4']):
            ch_norm = f'{ch_base}_norm'
            ch_raw = f'{ch_base}_raw'
            
            if ch_norm in self.fnirs_channels and len(self.fnirs_channels[ch_norm]) > 0:
                norm_data = np.array(list(self.fnirs_channels[ch_norm])[-100:])
                raw_data = np.array(list(self.fnirs_channels[ch_raw])[-100:])
                
                self.axes['info'].text(0.15, y_pos, f'{ch_base}:', fontsize=9,
                                     color=self.fnirs_colors[ch_base],
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.3, y_pos, f'{np.mean(norm_data):.3f}',
                                     fontsize=8, color='white',
                                     transform=self.axes['info'].transAxes)
                self.axes['info'].text(0.5, y_pos, f'({np.mean(raw_data):.1f})',
                                     fontsize=8, color='#aaaaaa',
                                     transform=self.axes['info'].transAxes)
                y_pos -= 0.035
        
        # System stats
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, 'System:', fontsize=10,
                             weight='bold', color='#95A5A6',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        self.axes['info'].text(0.15, y_pos, 'Packets:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.packet_count}',
                             fontsize=9, color='white',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.15, y_pos, 'EEG:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.eeg_packet_count}',
                             fontsize=9, color='#4ECDC4',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.04
        
        self.axes['info'].text(0.15, y_pos, 'fNIRS:', fontsize=9,
                             color='white',
                             transform=self.axes['info'].transAxes)
        self.axes['info'].text(0.4, y_pos, f'{self.fnirs_packet_count}',
                             fontsize=9, color='#E74C3C',
                             transform=self.axes['info'].transAxes)
    
    def start(self):
        """Start the visualizer"""
        print("\n" + "="*60)
        print("   MUSE S ATHENA - CONFUSION DETECTION TRAINING MODE")
        print("="*60)
        print(f"\n🧠 Listening for OSC data on UDP port {self.port}")
        print("\n📚 TRAINING MODE ACTIVE")
        print("\n🎯 CONFUSION MARKERS (use in teleprompter window):")
        print("  'c' = Word confusion (when a word is confusing)")
        print("  's' = Sentence confusion (when a sentence doesn't make sense)")
        print("  '1','2','3' = Other event markers")
        
        print("\n📖 TELEPROMPTER CONTROLS:")
        print("  ↑/↓ = Scroll text up/down")
        print("  ←/→ = Previous/Next text passage")
        print("  Space = Start/Stop recording")
        print("  +/- = Increase/decrease font size")
        
        print("\n🖥️ VISUALIZER CONTROLS:")
        print("  '+'/'-' = Increase/decrease time window")
        print("  'r' = Reset buffers")
        print("  'q' = Quit (saves data if recording)")
        
        print("\n💾 DATA COLLECTION:")
        print("  1. Teleprompter window will open automatically")
        print("  2. Click 'Begin Recording' or press Space to start")
        print("  3. Read the displayed text carefully")
        print("  4. Press 'c' when confused by a word")
        print("  5. Press 's' when a sentence is confusing")
        print("  6. Click 'Stop Recording' to save data")
        print("  7. Data auto-saves on exit if recording")
        
        print("\n🔬 DATA FORMAT:")
        print("  • EEG: 4 channels (TP9, AF7, AF8, TP10)")
        print("  • fNIRS: 8 values (4 normalized + 4 raw)")
        print("  • Events: Timestamped confusion markers")
        print("  • Text: Which passage was being read")
        
        # Start UDP receiver
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(0.1)
            self.running = True
            
            receiver_thread = threading.Thread(target=self.receiver_loop)
            receiver_thread.daemon = True
            receiver_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start receiver: {e}")
            return
        
        # Setup visualization
        self.setup_visualization()
        
        # Create and open teleprompter window
        print("\n📖 Opening teleprompter window...")
        self.teleprompter = TeleprompterWindow(self)
        
        # Start teleprompter update loop
        self.teleprompter.update_loop()
        
        # Keyboard shortcuts for main window
        def on_key(event):
            if event.key == 'q':
                print("\nQuitting...")
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.on_close()
                self.cleanup_on_exit()
                plt.close('all')
                self.stop()
            elif event.key == '+' or event.key == '=':
                self.window_duration = min(self.window_duration + 2, 30)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == '-':
                self.window_duration = max(self.window_duration - 2, 2)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == 'r':
                # Reset buffers
                with self.lock:
                    for ch in self.eeg_channels.values():
                        ch.clear()
                    for ch in self.fnirs_channels.values():
                        ch.clear()
                    for ch in self.motion_channels.values():
                        ch.clear()
                    for ch in self.ref_channels.values():
                        ch.clear()
                    self.timestamps.clear()
                print("Buffers reset")
            # Also handle confusion markers in main window
            elif event.key == 'c' or event.key == 'C':
                self.record_event('c')
            elif event.key == 's' or event.key == 'S':
                self.record_event('s')
            elif event.key == '1':
                self.record_event('1')
            elif event.key == '2':
                self.record_event('2')
            elif event.key == '3':
                self.record_event('3')
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Handle window close event
        def on_close(event):
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
            self.cleanup_on_exit()
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=40,  # 25 FPS
            blit=False,
            cache_frame_data=False
        )
        
        print("\n✅ Visualization started!")
        print("📖 Teleprompter window should be open - focus it to use controls")
        print("First 5 EEG and fNIRS packets will be printed for verification.")
        print("\n" + "="*60)
        
        try:
            # Run both windows
            def run_teleprompter():
                if self.teleprompter:
                    self.teleprompter.root.mainloop()
            
            teleprompter_thread = threading.Thread(target=run_teleprompter)
            teleprompter_thread.daemon = True
            teleprompter_thread.start()
            
            plt.show()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
            self.cleanup_on_exit()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Close teleprompter if still open
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.on_close()
        
        print(f"\n{'='*50}")
        print(f"VISUALIZER STOPPED")
        print(f"Total packets received: {self.packet_count}")
        print(f"EEG packets: {self.eeg_packet_count}")
        print(f"fNIRS packets: {self.fnirs_packet_count}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    # Create and start visualizer
    visualizer = MuseAthenaVisualizer(
        port=8052,
        buffer_size=2000,
        window_duration=10
    )
    
    try:
        visualizer.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()