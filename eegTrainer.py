#!/usr/bin/env python3
"""
eegTrainer.py:
Visualizer for Muse S Athena with EEG and fNIRS (optics) data
Modified for ML training with confusion detection using mouse selection
Saves data as NPZ files with event timestamps and selected text
Cursor tracking identifies which word is being read
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
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import os
from datetime import datetime
from tkinter import filedialog
import tkinter as tk
import signal as sig
import atexit
import sys

# Training texts
TRAINING_TEXTS = [
"""Epineural cuff electrodes. Epineural cuff electrodes are the simplest of nerve interface designs, usually containing 2 or more electrodes that are insulated and wrap around the surface of the epineurium of the peripheral nerve.""",

"""To date this type of interface is the only used in the clinic. This approach elicits a low FBR, making them quite stable for chronic implantation because the technology relies on compound signals to and from the nerve.""",
]

class TeleprompterWindow:
    """Teleprompter window with improved text selection for confusion marking"""
    def __init__(self, parent_visualizer):
        self.parent = parent_visualizer
        self.root = tk.Tk()
        self.root.title("👁 READING MATERIAL - Confusion Detection Training")
        
        # Window setup
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        self.active = True
        self.labeling_mode = False
        
        # Selection tracking
        self.selection_start = None
        self.selection_end = None
        self.is_dragging = False
        
        # Cursor tracking
        self.current_word = ""
        
        # Track previous word position for efficient tag removal
        self.current_word_start = None
        self.current_word_end = None
        
        # Pre-calculated word positions for FAST lookup (avoids slow Tkinter calls)
        self.word_positions = []  # [(char_start, char_end, word, tk_start, tk_end), ...]
        self.current_word_index = -1
        
        # Return sweep detection (when cursor moves back to start of next line)
        self.last_cursor_x = 0
        self.in_return_sweep = False
        self.return_sweep_threshold = 150  # pixels - leftward jump triggers return sweep
        
        # UI Setup
        self._setup_ui()
        
        # Bind events
        self._bind_events()
        
        # Initialize display
        self.update_display()
        self.track_cursor()
    
    def _setup_ui(self):
        """Setup the UI components"""
        # Header
        header_frame = tk.Frame(self.root, bg='#1a1a1a', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, 
                text="👁 CONFUSION DETECTION TRAINING",
                font=('Arial', 24, 'bold'),
                fg='#FFD93D',
                bg='#1a1a1a').pack(pady=10)
        
        tk.Label(header_frame,
                text="C: Toggle Labeling | LEFT-DRAG: multi-word confusion | RIGHT-CLICK: sentence confusion",
                font=('Arial', 14),
                fg='#4ECDC4',
                bg='#1a1a1a').pack()
        
        # Text display
        text_frame = tk.Frame(self.root, bg='#0a0a0a')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_display = tk.Text(text_frame,
                                   font=('Georgia', 28, 'normal'),
                                   bg='#0a0a0a',
                                   fg='white',
                                   wrap=tk.WORD,
                                   padx=40,
                                   pady=30,
                                   spacing1=10,
                                   spacing2=8,
                                   spacing3=10,
                                   insertwidth=0,
                                   highlightthickness=0,
                                   borderwidth=0,
                                   relief=tk.FLAT,
                                   cursor="hand2")
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.text_display.config(state=tk.DISABLED)
        
        # Configure selection colors
        self.text_display.tag_configure("selection", background="#4444ff", foreground="white")
        self.text_display.tag_configure("sentence_highlight", background="#664488", foreground="white")
        self.text_display.tag_configure("current_word", underline=True, foreground="#FFD93D")
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#1a1a1a', height=120)
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
                                        text="⏺ NOT RECORDING",
                                        font=('Arial', 16, 'bold'),
                                        fg='#888888',
                                        bg='#1a1a1a')
        self.recording_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.labeling_status = tk.Label(status_frame,
                                       text="📖 READING MODE",
                                       font=('Arial', 16, 'bold'),
                                       fg='#96CEB4',
                                       bg='#1a1a1a')
        self.labeling_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.event_status = tk.Label(status_frame,
                                    text="Events: Word=0, Sentence=0",
                                    font=('Arial', 16),
                                    fg='#E74C3C',
                                    bg='#1a1a1a')
        self.event_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Additional status
        self.word_status = tk.Label(status_frame,
                                   text="Current word: -",
                                   font=('Arial', 14, 'italic'),
                                   fg='#FFD93D',
                                   bg='#1a1a1a')
        self.word_status.pack(side=tk.RIGHT, padx=20, pady=5)
        
        tk.Label(status_frame,
                text="↑/↓: Scroll | ←/→: Change Text | Space: Record | C: Toggle Label | +/-: Font",
                font=('Arial', 12),
                fg='#888888',
                bg='#1a1a1a').pack(side=tk.BOTTOM, padx=20, pady=5)
        
        self.last_click_label = tk.Label(status_frame,
                                       text="Last marked: -",
                                       font=('Arial', 12),
                                       fg='#FF6B6B',
                                       bg='#1a1a1a')
        self.last_click_label.pack(side=tk.BOTTOM, padx=20, pady=2)
    
    def _bind_events(self):
        """Bind all event handlers"""
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Mouse events
        self.text_display.bind('<Motion>', self.on_mouse_motion)
        self.text_display.bind('<Leave>', self.on_mouse_leave)
        self.text_display.bind('<Button-1>', self.on_left_down)
        self.text_display.bind('<B1-Motion>', self.on_left_drag)
        self.text_display.bind('<ButtonRelease-1>', self.on_left_up)
        self.text_display.bind('<Button-3>', self.on_right_click)
        self.text_display.bind('<Button-2>', self.on_right_click)
    
    def _build_word_index(self):
        """Pre-calculate word positions for FAST cursor tracking.
        
        This eliminates 3 slow Tkinter calls per mouse motion by pre-computing
        word boundaries when text is loaded.
        """
        self.word_positions = []
        text = TRAINING_TEXTS[self.parent.current_text_index]
        
        i = 0
        while i < len(text):
            # Skip whitespace
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            
            # Find word end
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            
            word = text[start:i]
            # Pre-compute Tkinter indices (line 1 since no newlines in text)
            tk_start = f"1.{start}"
            tk_end = f"1.{i}"
            self.word_positions.append((start, i, word, tk_start, tk_end))
        
        self.current_word_index = -1
    
    def _find_word_at_char(self, char_pos):
        """Binary search to find word index at character position. O(log n)."""
        if not self.word_positions:
            return -1
        
        left, right = 0, len(self.word_positions) - 1
        while left <= right:
            mid = (left + right) // 2
            start, end = self.word_positions[mid][:2]
            if char_pos < start:
                right = mid - 1
            elif char_pos >= end:
                left = mid + 1
            else:
                return mid
        return -1
    
    def on_left_down(self, event):
        """Start text selection on left mouse down"""
        if not self.labeling_mode:
            return
            
        self.is_dragging = True
        self.selection_start = self.text_display.index(f"@{event.x},{event.y}")
        self.selection_end = self.selection_start
        
        # Clear existing selection
        self.text_display.tag_remove("selection", "1.0", tk.END)
    
    def on_left_drag(self, event):
        """Update selection during drag"""
        if not self.is_dragging or not self.labeling_mode:
            return
            
        # Update selection end point
        self.selection_end = self.text_display.index(f"@{event.x},{event.y}")
        
        # Update visual selection
        self.text_display.tag_remove("selection", "1.0", tk.END)
        self.text_display.tag_add("selection", self.selection_start, self.selection_end)
    
    def on_left_up(self, event):
        """Complete selection on mouse up"""
        if not self.is_dragging or not self.labeling_mode:
            return
            
        self.is_dragging = False
        
        if not self.parent.is_recording:
            self.text_display.tag_remove("selection", "1.0", tk.END)
            tk.messagebox.showinfo("Not Recording", "Start recording first before marking confusion events.")
            return
        
        # Get selected text with word expansion
        try:
            if self.text_display.compare(self.selection_start, "<", self.selection_end):
                start = self.selection_start
                end = self.selection_end
            else:
                start = self.selection_end
                end = self.selection_start
            
            # Expand selection to full word boundaries
            expanded_start = self.text_display.index(f"{start} wordstart")
            expanded_end = self.text_display.index(f"{end} wordend")
            
            # Get the expanded text
            selected_text = self.text_display.get(expanded_start, expanded_end).strip()
            
            if selected_text:
                # Update visual selection to show expanded range
                self.text_display.tag_remove("selection", "1.0", tk.END)
                self.text_display.tag_add("selection", expanded_start, expanded_end)
                
                # Extract individual words (filter out empty strings and punctuation-only)
                words = [w.strip() for w in selected_text.split() if w.strip() and any(c.isalnum() for c in w)]
                
                # Calculate character positions for unique identification
                full_text = self.text_display.get("1.0", tk.END)
                char_start = len(self.text_display.get("1.0", expanded_start))
                char_end = len(self.text_display.get("1.0", expanded_end))
                
                # Build event data with full info
                event_data = {
                    'text': selected_text,           # The full selected text
                    'words': words,                   # List of individual words
                    'char_start': char_start,         # Character offset from text start
                    'char_end': char_end,             # Character offset end
                    'word_count': len(words)
                }
                
                # Record the confusion event
                self.parent.record_event('word_confusion', event_data)
                self.flash_event("WORD", selected_text)
                
                # Display words in status
                words_preview = ', '.join(words[:3])
                if len(words) > 3:
                    words_preview += f'... ({len(words)} words)'
                self.last_click_label.config(text=f"Last marked: [{words_preview}] (word confusion)")
                
                # Clear selection after brief delay
                self.root.after(500, lambda: self.text_display.tag_remove("selection", "1.0", tk.END))
        except Exception as e:
            print(f"Error processing selection: {e}")
    
    def on_right_click(self, event):
        """Handle right click for sentence confusion"""
        if not self.labeling_mode:
            return
        
        if not self.parent.is_recording:
            tk.messagebox.showinfo("Not Recording", "Start recording first before marking confusion events.")
            return
        
        try:
            # Get click position
            click_pos = self.text_display.index(f"@{event.x},{event.y}")
            
            # Get entire text
            text_content = self.text_display.get("1.0", tk.END)
            
            # Find sentence boundaries
            click_offset = len(self.text_display.get("1.0", click_pos))
            
            # Find previous period or start
            prev_period = text_content.rfind('.', 0, click_offset)
            if prev_period == -1:
                prev_period = 0
            else:
                prev_period += 1  # Start after the period
            
            # Find next period or end
            next_period = text_content.find('.', click_offset)
            if next_period == -1:
                next_period = len(text_content) - 1
            else:
                next_period += 1  # Include the period
            
            # Extract sentence
            sentence = text_content[prev_period:next_period].strip()
            
            if sentence:
                # Convert character offsets back to Text widget indices for highlighting
                start_idx = self.text_display.index(f"1.0 + {prev_period} chars")
                end_idx = self.text_display.index(f"1.0 + {next_period} chars")
                
                # Highlight the sentence
                self.text_display.tag_remove("sentence_highlight", "1.0", tk.END)
                self.text_display.tag_add("sentence_highlight", start_idx, end_idx)
                
                # Record the event with position info
                event_data = {
                    'text': sentence,
                    'char_start': prev_period,
                    'char_end': next_period
                }
                self.parent.record_event('sentence_confusion', event_data)
                self.last_click_label.config(text=f"Last marked: '{sentence[:30]}...' (sentence confusion)")
                
                # Remove highlight after 2 seconds
                self.root.after(2000, lambda: self.text_display.tag_remove("sentence_highlight", "1.0", tk.END))
        except Exception as e:
            print(f"Error processing sentence: {e}")
    
    def on_mouse_motion(self, event):
        """Track cursor position over text with return sweep detection.
        
        OPTIMIZED: Uses pre-calculated word positions to reduce Tkinter calls
        from 4+ down to 1, making cursor tracking much more responsive.
        """
        cursor_x = event.x
        
        # Return sweep detection (always runs immediately)
        # Check if we just made a large leftward jump (return sweep started)
        if cursor_x < self.last_cursor_x - self.return_sweep_threshold:
            self.in_return_sweep = True
        
        # Exit return sweep on any rightward movement
        if self.in_return_sweep and cursor_x > self.last_cursor_x:
            self.in_return_sweep = False
        
        self.last_cursor_x = cursor_x
        
        # Don't update current word during return sweep
        if self.in_return_sweep:
            return
        
        # FAST word lookup using pre-calculated positions
        try:
            # Single Tkinter call to get character position
            index = self.text_display.index(f"@{event.x},{event.y}")
            
            # Parse "line.column" to get character offset (fast string op)
            col = int(index.split('.')[1])
            
            # Binary search for word (pure Python, O(log n), very fast)
            word_idx = self._find_word_at_char(col)
            
            # Only update if we're on a different word
            if word_idx != self.current_word_index and word_idx >= 0:
                _, _, word, tk_start, tk_end = self.word_positions[word_idx]
                
                self.current_word = word
                self.parent.current_word = word
                
                # Update underline: remove old, add new
                if self.current_word_start and self.current_word_end:
                    self.text_display.tag_remove("current_word",
                                                  self.current_word_start,
                                                  self.current_word_end)
                
                self.text_display.tag_add("current_word", tk_start, tk_end)
                self.current_word_start = tk_start
                self.current_word_end = tk_end
                self.current_word_index = word_idx
                
                # Update status label
                self.word_status.config(text=f"Current word: {word}")
        except:
            pass
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving text area"""
        self.current_word = ""
        self.parent.current_word = ""
        self.word_status.config(text="Current word: -")
        # Remove underline efficiently from tracked position
        if self.current_word_start and self.current_word_end:
            self.text_display.tag_remove("current_word", 
                                          self.current_word_start, 
                                          self.current_word_end)
        self.current_word_start = None
        self.current_word_end = None
        self.current_word_index = -1
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        key_actions = {
            'Up': lambda: self.scroll_text(-0.05),
            'Down': lambda: self.scroll_text(0.05),
            'Left': lambda: (self.parent.previous_text(), self.update_display()),
            'Right': lambda: (self.parent.next_text(), self.update_display()),
            'space': lambda: (self.parent.toggle_recording(), self.update_status()),
        }
        
        if event.keysym in key_actions:
            key_actions[event.keysym]()
        elif event.char.lower() == 'q':
            self.on_close()
        elif event.char.lower() == 'c':
            self.toggle_labeling_mode()
        elif event.char in ['+', '=']:
            self.adjust_font_size(2)
        elif event.char == '-':
            self.adjust_font_size(-2)
    
    def adjust_font_size(self, delta):
        """Adjust text display font size"""
        current_font = self.text_display.cget('font')
        if isinstance(current_font, str):
            font_parts = current_font.split()
            current_size = int(font_parts[1]) if len(font_parts) > 1 else 28
        else:
            current_size = 28
        new_size = max(16, min(current_size + delta, 48))
        self.text_display.config(font=('Georgia', new_size, 'normal'))
    
    def flash_event(self, event_type, text):
        """Visual feedback for event recording"""
        original_bg = self.text_display.cget('bg')
        flash_color = '#2a2a2a' if event_type == "WORD" else '#1a2a2a'
        self.text_display.config(bg=flash_color)
        self.root.after(100, lambda: self.text_display.config(bg=original_bg))
    
    def toggle_labeling_mode(self):
        """Toggle between reading and labeling modes"""
        self.labeling_mode = not self.labeling_mode
        
        if self.labeling_mode:
            self.parent.pause_data_collection()
            self.labeling_status.config(text="🏷️ LABELING MODE", fg='#ff6666')
            self.text_display.config(cursor="crosshair")
            print("\n🏷️ LABELING MODE: Click and drag to select confusing text.")
        else:
            self.parent.resume_data_collection()
            self.labeling_status.config(text="📖 READING MODE", fg='#96CEB4')
            self.text_display.config(cursor="hand2")
            self.text_display.tag_remove("selection", "1.0", tk.END)
            print("\n📖 READING MODE: Data collection resumed.")
    
    def scroll_text(self, amount):
        """Scroll the text display"""
        self.text_display.yview_scroll(int(amount * 10), "units")
    
    def update_display(self):
        """Update text display with current passage"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', TRAINING_TEXTS[self.parent.current_text_index])
        self.text_display.config(state=tk.DISABLED)
        self.text_display.yview_moveto(0)
        self.text_status.config(text=f"Text: {self.parent.current_text_index + 1}/{len(TRAINING_TEXTS)}")
        self.last_click_label.config(text="Last marked: -")
        # Reset current word tracking
        self.current_word = ""
        self.parent.current_word = ""
        self.word_status.config(text="Current word: -")
        self.current_word_start = None
        self.current_word_end = None
        self.current_word_index = -1
        # Pre-calculate word positions for fast cursor tracking
        self._build_word_index()
    
    def update_status(self):
        """Update recording and event status"""
        if self.parent.is_recording:
            elapsed = time.time() - self.parent.recording_start_time
            status_text = f"⏺ RECORDING: {elapsed:.1f}s"
            if self.parent.data_collection_paused:
                status_text += " (PAUSED)"
            self.recording_status.config(text=status_text, fg='#ff4444')
            
            # Count events
            word_count = sum(1 for _, event_type, _ in self.parent.recorded_events 
                           if event_type == 'word_confusion')
            sentence_count = sum(1 for _, event_type, _ in self.parent.recorded_events 
                              if event_type == 'sentence_confusion')
            
            self.event_status.config(text=f"Events: Word={word_count}, Sentence={sentence_count}")
        else:
            self.recording_status.config(text="⏺ NOT RECORDING", fg='#888888')
    
    def track_cursor(self):
        """Backup cursor tracking (Motion events handle most cases, this is fallback)"""
        if self.active:
            try:
                x, y = self.text_display.winfo_pointerxy()
                widget_x = self.text_display.winfo_rootx()
                widget_y = self.text_display.winfo_rooty()
                rel_x = x - widget_x
                rel_y = y - widget_y
                
                if (0 <= rel_x <= self.text_display.winfo_width() and 
                    0 <= rel_y <= self.text_display.winfo_height()):
                    event = type('obj', (object,), {'x': rel_x, 'y': rel_y})
                    self.on_mouse_motion(event)
            except:
                pass
            
            # Slower poll since <Motion> events handle most tracking
            self.root.after(100, self.track_cursor)
    
    def on_close(self):
        """Clean window close"""
        self.active = False
        self.root.destroy()
    
    def update_loop(self):
        """Regular status update loop"""
        if self.active:
            self.update_status()
            self.root.after(100, self.update_loop)


class MuseAthenaVisualizer:
    def __init__(self, port=8052, buffer_size=2000, window_duration=10):
        # Network
        self.port = port
        self.socket = None
        self.running = False
        
        # Data buffers
        self.buffer_size = buffer_size
        self.window_duration = window_duration
        self.timestamps = deque(maxlen=buffer_size)
        
        # Channel storage
        self.eeg_channels = {ch: deque(maxlen=buffer_size) 
                            for ch in ['TP9', 'AF7', 'AF8', 'TP10']}
        self.fnirs_channels = {f'Ch{i}_{t}': deque(maxlen=buffer_size) 
                              for i in range(1,5) for t in ['norm', 'raw']}
        self.motion_channels = {ch: deque(maxlen=buffer_size) 
                               for ch in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']}
        self.ref_channels = {ch: deque(maxlen=buffer_size) for ch in ['DRL', 'REF']}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Visualization
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.spectral_lines = {}
        
        # UI elements
        self.teleprompter = None
        self.current_text_index = 0
        self.current_word = ""
        self.data_collection_paused = False
        
        # Recording
        self.is_recording = False
        self.recording_start_time = None
        self.record_button = None
        self.recorded_timestamps = []
        self.recorded_eeg = []
        self.recorded_fnirs = []
        self.recorded_motion = []
        self.recorded_ref = []
        self.recorded_events = []
        self.recorded_words = []
        
        # Last values for interpolation
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Cleanup handlers
        self.shutting_down = False
        atexit.register(self.cleanup_on_exit)
        sig.signal(sig.SIGINT, self.signal_handler)
        
        # Spectral parameters
        self.sample_rate = 256
        self.spectral_window_size = 512
        self.max_freq = 70
        self.freq_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        # Colors
        self.eeg_colors = {'TP9': '#FF6B6B', 'AF7': '#4ECDC4', 
                          'AF8': '#45B7D1', 'TP10': '#96CEB4'}
        self.fnirs_colors = {f'Ch{i}': c for i, c in 
                            zip(range(1,5), ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])}
    
    def parse_osc_message(self, data):
        """Parse OSC message from binary data"""
        try:
            def parse_string(data, offset):
                end = data.find(b'\x00', offset)
                if end == -1:
                    return None, offset
                string = data[offset:end].decode('ascii')
                offset = ((end + 4) // 4) * 4
                return string, offset
            
            offset = 0
            address, offset = parse_string(data, offset)
            if not address:
                return None
            
            if not address.startswith('/'):
                address = '/' + address
            
            type_tags, offset = parse_string(data, offset)
            if not type_tags or not type_tags.startswith(','):
                return None
            
            type_tags = type_tags[1:]
            args = []
            
            for tag in type_tags:
                if tag == 'f' and offset + 4 <= len(data):
                    args.append(struct.unpack('>f', data[offset:offset+4])[0])
                    offset += 4
                elif tag == 'i' and offset + 4 <= len(data):
                    args.append(struct.unpack('>i', data[offset:offset+4])[0])
                    offset += 4
            
            return {'address': address, 'args': args}
        except:
            return None
    
    def process_osc_message(self, message):
        """Process incoming OSC message"""
        address = message['address']
        args = message['args']
        timestamp = time.time()
        
        with self.lock:
            parts = address.strip('/').split('/')
            if len(parts) >= 2:
                data_type = parts[1]
                
                if data_type == 'eeg' and len(args) == 4:
                    self.timestamps.append(timestamp)
                    for ch, val in zip(['TP9', 'AF7', 'AF8', 'TP10'], args):
                        self.eeg_channels[ch].append(val)
                    self.eeg_packet_count += 1
                    self.last_eeg_data = args
                    
                    if self.is_recording and not self.data_collection_paused:
                        self.recorded_timestamps.append(timestamp)
                        self.recorded_eeg.append(args)
                        self.recorded_fnirs.append(self.last_fnirs_data if self.last_fnirs_data else [np.nan] * 8)
                        self.recorded_motion.append(self.last_motion_data if self.last_motion_data else [np.nan] * 6)
                        self.recorded_ref.append(self.last_ref_data if self.last_ref_data else [np.nan] * 2)
                        self.recorded_words.append(self.current_word)
                    
                    if self.eeg_packet_count <= 5:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                elif data_type == 'optics' and len(args) == 8:
                    for i, ch in enumerate([f'Ch{j}_{t}' for j in range(1,5) for t in ['norm', 'raw']]):
                        self.fnirs_channels[ch].append(args[i])
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                    
                    if self.fnirs_packet_count <= 5:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}, raw={args[4:]}")
                
                elif data_type == 'acc' and len(args) == 3:
                    for ch, val in zip(['acc_x', 'acc_y', 'acc_z'], args):
                        self.motion_channels[ch].append(val)
                    if not self.last_motion_data:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[:3] = args
                
                elif data_type == 'gyro' and len(args) == 3:
                    for ch, val in zip(['gyro_x', 'gyro_y', 'gyro_z'], args):
                        self.motion_channels[ch].append(val)
                    if not self.last_motion_data:
                        self.last_motion_data = [0, 0, 0, 0, 0, 0]
                    self.last_motion_data[3:] = args
                
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
                message = self.parse_osc_message(data)
                if message:
                    self.process_osc_message(message)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}")
    
    def next_text(self):
        """Navigate to next text"""
        self.current_text_index = (self.current_text_index + 1) % len(TRAINING_TEXTS)
        print(f"\n📖 Text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
    
    def previous_text(self):
        """Navigate to previous text"""
        self.current_text_index = (self.current_text_index - 1) % len(TRAINING_TEXTS)
        print(f"\n📖 Text {self.current_text_index + 1}/{len(TRAINING_TEXTS)}")
    
    def pause_data_collection(self):
        """Pause data collection during labeling"""
        self.data_collection_paused = True
    
    def resume_data_collection(self):
        """Resume data collection"""
        self.data_collection_paused = False
    
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
                self.recorded_words = []
            
            if self.record_button:
                self.record_button.label.set_text('Stop Recording')
                self.record_button.color = '#ff4444'
                self.record_button.hovercolor = '#ff6666'
            
            print(f"\n{'='*50}")
            print(f"RECORDING STARTED at {datetime.fromtimestamp(self.recording_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
        else:
            # Stop recording
            self.is_recording = False
            
            # Save data
            with self.lock:
                save_data = {
                    'timestamps': self.recorded_timestamps.copy(),
                    'eeg': self.recorded_eeg.copy(),
                    'fnirs': self.recorded_fnirs.copy(),
                    'motion': self.recorded_motion.copy(),
                    'ref': self.recorded_ref.copy(),
                    'events': self.recorded_events.copy(),
                    'words': self.recorded_words.copy(),
                    'start_time': self.recording_start_time,
                    'current_text_index': self.current_text_index
                }
                data_points = len(self.recorded_timestamps)
                events_count = len(self.recorded_events)
                unique_words = len(set(w for w in self.recorded_words if w))
            
            if self.record_button:
                self.record_button.label.set_text('Saving...')
                self.record_button.color = '#888888'
            
            duration = time.time() - self.recording_start_time
            print(f"\n{'='*50}")
            print(f"RECORDING STOPPED")
            print(f"Duration: {duration:.1f}s, Points: {data_points}, Events: {events_count}")
            print(f"{'='*50}")
            
            # Save in thread
            save_thread = threading.Thread(target=self._save_recording_thread, args=(save_data,))
            save_thread.daemon = True
            save_thread.start()
    
    def record_event(self, event_type, event_data):
        """Record confusion event with structured data"""
        if self.is_recording:
            timestamp = time.time()
            relative_time = timestamp - self.recording_start_time
            self.recorded_events.append((timestamp, event_type, event_data))
            
            if event_type == 'word_confusion':
                words = event_data.get('words', [])
                text = event_data.get('text', '')[:50]
                char_pos = event_data.get('char_start', 0)
                print(f"🤔 WORD confusion at {relative_time:.2f}s (pos {char_pos}): {words}")
            elif event_type == 'sentence_confusion':
                text = event_data.get('text', '')[:50]
                char_pos = event_data.get('char_start', 0)
                print(f"📄 SENTENCE confusion at {relative_time:.2f}s (pos {char_pos}): '{text}...'")
    
    def _save_recording_thread(self, save_data):
        """Save recording data in thread (auto-save, no dialog)"""
        try:
            # Auto-generate filename (no dialog - dialogs crash when called from background thread on macOS)
            save_dir = os.path.expanduser("~/Downloads/EEGRecordings")
            os.makedirs(save_dir, exist_ok=True)
            
            # Create descriptive filename with timestamp and event count
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            event_count = len(save_data['events'])
            duration = int(save_data['timestamps'][-1] - save_data['timestamps'][0]) if save_data['timestamps'] else 0
            filename = os.path.join(save_dir, f"eeg_confusion_{timestamp_str}_{duration}s_{event_count}events.npz")
            
            print(f"\n💾 Auto-saving to: {filename}")
            print("Converting data...")
            
            # Convert to numpy arrays
            timestamps = np.array(save_data['timestamps'])
            eeg_data = np.array(save_data['eeg']) if save_data['eeg'] else np.array([])
            fnirs_data = np.array(save_data['fnirs']) if save_data['fnirs'] else np.array([])
            motion_data = np.array(save_data['motion']) if save_data['motion'] else np.array([])
            ref_data = np.array(save_data['ref']) if save_data['ref'] else np.array([])
            words_array = np.array(save_data['words'], dtype=object) if save_data['words'] else np.array([], dtype=object)
            
            # Events - now with structured data
            if save_data['events']:
                event_timestamps = np.array([e[0] for e in save_data['events']])
                event_types = np.array([e[1] for e in save_data['events']])
                # Store full event data dictionaries (contains text, words, positions)
                event_data_list = np.array([e[2] for e in save_data['events']], dtype=object)
                # Also extract just the text for backward compatibility
                event_texts = np.array([e[2].get('text', '') if isinstance(e[2], dict) else e[2] 
                                       for e in save_data['events']], dtype=object)
                # Extract word lists (for word_confusion events)
                event_words_list = np.array([e[2].get('words', []) if isinstance(e[2], dict) else [] 
                                            for e in save_data['events']], dtype=object)
                # Extract character positions for unique identification
                event_char_starts = np.array([e[2].get('char_start', -1) if isinstance(e[2], dict) else -1 
                                             for e in save_data['events']])
                event_char_ends = np.array([e[2].get('char_end', -1) if isinstance(e[2], dict) else -1 
                                           for e in save_data['events']])
            else:
                event_timestamps = np.array([])
                event_types = np.array([])
                event_data_list = np.array([], dtype=object)
                event_texts = np.array([], dtype=object)
                event_words_list = np.array([], dtype=object)
                event_char_starts = np.array([])
                event_char_ends = np.array([])
            
            relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
            
            # Metadata
            metadata = {
                'device': 'Muse S Athena',
                'start_time': float(save_data['start_time']),
                'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps),
                'total_events': len(save_data['events']),
                'unique_words_tracked': len(set(w for w in save_data['words'] if w)),
                'text_passage_index': save_data['current_text_index'],
                'text_passage': TRAINING_TEXTS[save_data['current_text_index']],
                'eeg_channels': ['TP9', 'AF7', 'AF8', 'TP10'],
                'fnirs_channels': ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm', 
                                 'Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw'],
                'motion_channels': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                'ref_channels': ['DRL', 'REF'],
                'event_types': ['word_confusion', 'sentence_confusion', 'marker_1', 'marker_2', 'marker_3']
            }
            
            print("Saving to file...")
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
                event_texts=event_texts,
                event_words_list=event_words_list,
                event_char_starts=event_char_starts,
                event_char_ends=event_char_ends,
                event_data=event_data_list,
                words=words_array,
                metadata=metadata
            )
            
            print(f"\n✅ DATA SAVED: {filename}")
            print(f"📁 Location: {save_dir}")
            print(f"📊 Size: {os.path.getsize(filename) / 1024:.1f} KB")
        
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.record_button:
                self.record_button.label.set_text('Begin Recording')
                self.record_button.color = '#2a2a2a'
                self.record_button.hovercolor = '#3a3a3a'
                plt.draw()
    
    def save_recording_npz(self, auto_save=False):
        """Auto-save function for cleanup (called on exit if recording)"""
        if len(self.recorded_timestamps) == 0:
            return
        
        # Prepare data
        save_data = {
            'timestamps': self.recorded_timestamps,
            'eeg': self.recorded_eeg,
            'fnirs': self.recorded_fnirs,
            'motion': self.recorded_motion,
            'ref': self.recorded_ref,
            'events': self.recorded_events,
            'words': self.recorded_words,
            'start_time': self.recording_start_time,
            'current_text_index': self.current_text_index
        }
        
        # Auto-save to same directory as regular saves
        save_dir = os.path.expanduser("~/Downloads/EEGRecordings")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create descriptive filename
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        event_count = len(save_data['events'])
        duration = int(save_data['timestamps'][-1] - save_data['timestamps'][0]) if save_data['timestamps'] else 0
        filename = os.path.join(save_dir, f"eeg_confusion_AUTOSAVE_{timestamp_str}_{duration}s_{event_count}events.npz")
        
        print(f"Auto-saving to: {filename}")
        
        # Quick save without conversion
        try:
            # Convert to arrays
            timestamps = np.array(save_data['timestamps'])
            eeg_data = np.array(save_data['eeg']) if save_data['eeg'] else np.array([])
            fnirs_data = np.array(save_data['fnirs']) if save_data['fnirs'] else np.array([])
            motion_data = np.array(save_data['motion']) if save_data['motion'] else np.array([])
            ref_data = np.array(save_data['ref']) if save_data['ref'] else np.array([])
            words_array = np.array(save_data['words'], dtype=object) if save_data['words'] else np.array([], dtype=object)
            
            # Events - with structured data
            if save_data['events']:
                event_timestamps = np.array([e[0] for e in save_data['events']])
                event_types = np.array([e[1] for e in save_data['events']])
                event_data_list = np.array([e[2] for e in save_data['events']], dtype=object)
                event_texts = np.array([e[2].get('text', '') if isinstance(e[2], dict) else e[2] 
                                       for e in save_data['events']], dtype=object)
                event_words_list = np.array([e[2].get('words', []) if isinstance(e[2], dict) else [] 
                                            for e in save_data['events']], dtype=object)
                event_char_starts = np.array([e[2].get('char_start', -1) if isinstance(e[2], dict) else -1 
                                             for e in save_data['events']])
                event_char_ends = np.array([e[2].get('char_end', -1) if isinstance(e[2], dict) else -1 
                                           for e in save_data['events']])
            else:
                event_timestamps = np.array([])
                event_types = np.array([])
                event_data_list = np.array([], dtype=object)
                event_texts = np.array([], dtype=object)
                event_words_list = np.array([], dtype=object)
                event_char_starts = np.array([])
                event_char_ends = np.array([])
            
            relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
            
            metadata = {
                'device': 'Muse S Athena',
                'start_time': float(save_data['start_time']),
                'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps),
                'total_events': len(save_data['events']),
                'unique_words_tracked': len(set(w for w in save_data['words'] if w)),
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
                event_texts=event_texts,
                event_words_list=event_words_list,
                event_char_starts=event_char_starts,
                event_char_ends=event_char_ends,
                event_data=event_data_list,
                words=words_array,
                metadata=metadata
            )
            
            print(f"Auto-save complete: {filename}")
        except Exception as e:
            print(f"Auto-save error: {e}")
    
    def cleanup_on_exit(self):
        """Clean exit handler"""
        if self.shutting_down:
            return
        self.shutting_down = True
        
        if self.is_recording and len(self.recorded_timestamps) > 0:
            print("\nRecording in progress. Auto-saving...")
            self.save_recording_npz(auto_save=True)
        
        self.running = False
        if self.socket:
            self.socket.close()
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n\nReceived interrupt. Saving if recording...")
        self.cleanup_on_exit()
        sys.exit(0)
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create grid
        gs = GridSpec(6, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        # Create axes
        self.axes = {
            'eeg': self.fig.add_subplot(gs[0, 0]),
            'spectral': self.fig.add_subplot(gs[1, 0]),
            'fnirs': self.fig.add_subplot(gs[2, 0]),
            'motion': self.fig.add_subplot(gs[3, 0]),
            'gyro': self.fig.add_subplot(gs[4, 0]),
            'ref': self.fig.add_subplot(gs[5, 0]),
            'info': self.fig.add_subplot(gs[:5, 1])
        }
        
        # Configure axes
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name != 'info':
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Titles
        titles = {
            'eeg': ('EEG Channels (4 channels)', '#4ECDC4'),
            'spectral': ('Spectral Analysis - Power Spectral Density', '#FFD93D'),
            'fnirs': ('fNIRS/Optics - Functional Near-Infrared Spectroscopy', '#E74C3C'),
            'motion': ('Accelerometer', '#96CEB4'),
            'gyro': ('Gyroscope', '#9B59B6'),
            'ref': ('Reference Electrodes', '#95A5A6')
        }
        
        for ax_name, (title, color) in titles.items():
            self.axes[ax_name].set_title(title, fontsize=14, color=color, pad=10)
        
        # Labels
        self.axes['eeg'].set_ylabel('Amplitude (µV)')
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        self.axes['fnirs'].set_ylabel('Intensity')
        self.axes['motion'].set_ylabel('Acceleration (g)')
        self.axes['gyro'].set_ylabel('Angular velocity (°/s)')
        self.axes['ref'].set_ylabel('Voltage')
        self.axes['ref'].set_xlabel('Time (s)')
        
        # Info panel
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
        
        # Initialize lines
        self._initialize_lines()
        
        plt.tight_layout()
    
    def _initialize_lines(self):
        """Initialize plot lines"""
        # EEG lines
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95)
            self.lines[f'eeg_{ch}'] = line
        
        # Spectral lines
        self.spectral_lines = {}
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color, 
                                              linewidth=1.5, alpha=0.9)
            self.spectral_lines[ch] = line
        
        self.axes['spectral'].set_xlim(0, self.max_freq)
        
        # Add frequency band labels
        for band_name, (low, high) in self.freq_bands.items():
            self.axes['spectral'].axvline(x=low, color='white', linestyle=':', 
                                         alpha=0.3, linewidth=0.5)
            mid_freq = (low + high) / 2
            if mid_freq < self.max_freq:
                self.axes['spectral'].text(mid_freq, 0.98, band_name, 
                                         fontsize=8, color='white', alpha=0.7,
                                         ha='center', va='top',
                                         transform=self.axes['spectral'].get_xaxis_transform())
        
        # fNIRS lines
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color, linewidth=1.5, alpha=0.9)
            self.lines[f'fnirs_{ch_norm}'] = line
        
        # Motion lines
        colors = {
            'acc': ['#3498DB', '#2ECC71', '#9B59B6'],
            'gyro': ['#F39C12', '#E67E22', '#D35400']
        }
        
        for prefix, ax_name in [('acc', 'motion'), ('gyro', 'gyro')]:
            for i, axis in enumerate(['x', 'y', 'z']):
                ch = f'{prefix}_{axis}'
                line, = self.axes[ax_name].plot([], [], label=axis,
                                               color=colors[prefix][i],
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
        for ax_name in ['eeg', 'spectral', 'fnirs', 'motion', 'gyro', 'ref']:
            self.axes[ax_name].legend(loc='upper right', fontsize=8, 
                                     ncol=4 if ax_name in ['eeg', 'spectral'] else 3,
                                     framealpha=0.5)
    
    def compute_spectrum(self, data, sample_rate=256):
        """Compute power spectral density"""
        if len(data) < self.spectral_window_size:
            return None, None
        
        frequencies, psd = signal.welch(
            data, 
            fs=sample_rate, 
            nperseg=min(len(data), self.spectral_window_size),
            noverlap=min(len(data)//2, self.spectral_window_size//2),
            scaling='density'
        )
        
        freq_mask = frequencies <= self.max_freq
        frequencies = frequencies[freq_mask]
        psd = psd[freq_mask]
        psd_db = 10 * np.log10(psd + 1e-10)
        
        return frequencies, psd_db
    
    def update_plot(self, frame):
        """Update all plots"""
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Get minimum length for alignment
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Time axis
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                time_axis = timestamps - timestamps[-1]
                display_mask = time_axis >= -self.window_duration
                display_samples = np.sum(display_mask)
            else:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            # Update EEG
            filtered_eeg_data = {}
            eeg_values_for_scaling = []
            
            for ch_name in self.eeg_channels:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        # Get and filter data
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
                        # High-pass filter
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
                        
                        filtered_eeg_data[ch_name] = filtered_data
                        
                        # Update line
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            # Update spectral analysis
            if len(filtered_eeg_data) == 4:
                all_psd_values = []
                
                for ch_name, data in filtered_eeg_data.items():
                    if ch_name in self.spectral_lines:
                        frequencies, psd = self.compute_spectrum(data[-int(self.sample_rate * 4):])
                        
                        if frequencies is not None:
                            self.spectral_lines[ch_name].set_data(frequencies, psd)
                            all_psd_values.extend(psd)
                
                if all_psd_values:
                    y_min = np.percentile(all_psd_values, 5) - 5
                    y_max = np.percentile(all_psd_values, 95) + 5
                    self.axes['spectral'].set_ylim(y_min, y_max)
            
            # Update other channels (fNIRS, motion, ref)
            for channel_dict, prefix in [(self.fnirs_channels, 'fnirs'), 
                                        (self.motion_channels, 'motion'),
                                        (self.ref_channels, 'ref')]:
                for ch_name, data_deque in channel_dict.items():
                    if len(data_deque) > 0:
                        # Determine correct line key
                        if prefix == 'fnirs' and ch_name.endswith('_norm'):
                            line_key = f'{prefix}_{ch_name}'
                        elif prefix == 'fnirs' and ch_name.endswith('_raw'):
                            continue  # Skip raw channels for display
                        else:
                            line_key = f'{prefix}_{ch_name}'
                        
                        if line_key in self.lines:
                            data_array = np.array(list(data_deque))
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
            
            # Special EEG scaling
            if eeg_values_for_scaling:
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                y_range = 4 * eeg_std
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
            
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
            status_text = f'⏺ REC: {elapsed:.1f}s'
            if self.data_collection_paused:
                status_text += ' (PAUSED)'
            self.axes['info'].text(0.1, y_pos, status_text,
                                 fontsize=10, color='#ff4444', weight='bold',
                                 transform=self.axes['info'].transAxes)
            
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'Samples: {len(self.recorded_timestamps)}',
                                 fontsize=9, color='#ff6666',
                                 transform=self.axes['info'].transAxes)
            
            # Event counts
            y_pos -= 0.04
            word_confusion = sum(1 for _, event_type, _ in self.recorded_events 
                                if event_type == 'word_confusion')
            sentence_confusion = sum(1 for _, event_type, _ in self.recorded_events 
                                   if event_type == 'sentence_confusion')
            
            self.axes['info'].text(0.1, y_pos, f'🤔 Word: {word_confusion}', 
                                 fontsize=9, color='#ffaa44',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
            self.axes['info'].text(0.1, y_pos, f'📄 Sentence: {sentence_confusion}', 
                                 fontsize=9, color='#ff8844',
                                 transform=self.axes['info'].transAxes)
        
        # Current word
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, f'Word: {self.current_word[:15] if self.current_word else "-"}', 
                             fontsize=9, color='#FFD93D',
                             transform=self.axes['info'].transAxes)
        
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
        
        for ch in self.eeg_channels:
            if len(self.eeg_channels[ch]) > 0:
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
        
        # System stats
        y_pos -= 0.06
        self.axes['info'].text(0.1, y_pos, 'System:', fontsize=10,
                             weight='bold', color='#95A5A6',
                             transform=self.axes['info'].transAxes)
        y_pos -= 0.05
        
        stats = [
            ('Packets:', self.packet_count),
            ('EEG:', self.eeg_packet_count),
            ('fNIRS:', self.fnirs_packet_count)
        ]
        
        for label, value in stats:
            self.axes['info'].text(0.15, y_pos, label, fontsize=9, color='white',
                                 transform=self.axes['info'].transAxes)
            self.axes['info'].text(0.4, y_pos, f'{value}', fontsize=9, 
                                 color='#4ECDC4' if 'EEG' in label else '#E74C3C' if 'fNIRS' in label else 'white',
                                 transform=self.axes['info'].transAxes)
            y_pos -= 0.04
    
    def start(self):
        """Start the visualizer"""
        print("\n" + "="*60)
        print("   MUSE S ATHENA - CONFUSION DETECTION WITH TEXT SELECTION")
        print("="*60)
        print(f"\n📡 Listening for OSC data on UDP port {self.port}")
        print("\n🖱️ IMPROVED SELECTION SYSTEM:")
        print("  • LEFT-DRAG: Select multi-word phrases that confuse you")
        print("  • RIGHT-CLICK: Mark entire sentence as confusing")
        print("  • C: Toggle between reading and labeling modes")
        print("\n📊 Data saves with full text selections for better ML training")
        
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
        
        # Create teleprompter on main thread (required by macOS)
        print("\n🖥️ Opening teleprompter window...")
        self.teleprompter = TeleprompterWindow(self)
        self.teleprompter.update_loop()
        
        # Keyboard shortcuts
        def on_key(event):
            if event.key == 'q':
                print("\nQuitting...")
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.on_close()
                self.cleanup_on_exit()
                plt.close('all')
                self.stop()
            elif event.key in ['+', '=']:
                self.window_duration = min(self.window_duration + 2, 30)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == '-':
                self.window_duration = max(self.window_duration - 2, 2)
                print(f"Window duration: {self.window_duration}s")
            elif event.key == 'r':
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
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Close handler
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
        print("🖥️ Teleprompter window should be open - focus it to use controls")
        print("\n" + "="*60)
        
        try:
            # On macOS, both Tkinter and matplotlib must run on the main thread.
            # TkAgg backend allows them to share the event loop.
            # We use plt.show(block=False) and let Tkinter drive updates.
            
            # Schedule periodic matplotlib canvas updates via Tkinter
            def update_matplotlib():
                try:
                    if self.fig and plt.fignum_exists(self.fig.number):
                        self.fig.canvas.draw_idle()
                        self.fig.canvas.flush_events()
                except:
                    pass
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.root.after(40, update_matplotlib)  # ~25 FPS
            
            # Show matplotlib non-blocking
            plt.show(block=False)
            
            # Start matplotlib updates via Tkinter
            update_matplotlib()
            
            # Run Tkinter mainloop (this drives everything on macOS)
            self.teleprompter.root.mainloop()
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
        
        if self.teleprompter and self.teleprompter.active:
            self.teleprompter.on_close()
        
        print(f"\n{'='*50}")
        print(f"VISUALIZER STOPPED")
        print(f"Total packets: {self.packet_count}")
        print(f"EEG: {self.eeg_packet_count}, fNIRS: {self.fnirs_packet_count}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    visualizer = MuseAthenaVisualizer(port=8052, buffer_size=2000, window_duration=10)
    try:
        visualizer.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()