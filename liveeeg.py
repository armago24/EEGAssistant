#!/usr/bin/env python3
"""
liveeeg.py:
Live neuroadaptive reader with real-time confusion prediction from EEG/fNIRS

Based on eegTrainer.py but modified for INFERENCE instead of training:
- Loads a pre-trained RNN model for confusion detection
- Makes predictions in real-time as you read
- Press 'H' to highlight predicted confusing words in the current paragraph
- No manual labeling required - the AI predicts for you!

Usage:
    python3 liveeeg.py --model confusion_rnn_best.pth

Make sure to train the model first using train_confusion_rnn.py
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
from tkinter import messagebox
import signal as sig
import atexit
import sys
import argparse
import torch

# Import our confusion detection components
from confusion_rnn import (
    ConfusionDetectorRNN, 
    ConfusionDetectorGRU, 
    EEGPreprocessor,
    LiveConfusionPredictor
)

# Training texts (same as eegTrainer.py for consistency)
TRAINING_TEXTS = [
"""Developments in microfabrication technology have enabled the production of neural electrode arrays with hundreds of closely spaced recording sites, and electrodes with thousands of sites are under development. These probes in principle allow the simultaneous recording of very large numbers of neurons. However, use of this technology requires the development of techniques for decoding the spike times of the recorded neurons from the raw data captured from the probes. Here we present a set of tools to solve this problem, implemented in a suite of practical, user-friendly, open-source software. We validate these methods on data from the cortex, hippocampus and thalamus of rat, mouse, macaque and marmoset, demonstrating error rates as low as 5%.""",

"""One of the most powerful techniques for neuronal population recording is extracellular electrophysiology using microfabricated electrode arrays 1–3. Advances in microfabrication have continually increased the number of recording sites available on neural probes, and the number of recordable neurons is further increased by having closely spaced recording sites. Indeed, while a single sharp electrode can provide good isolation of one or two neurons, placing as few as four recording sites together in a tetrode can reveal the firing patterns of 10–20 simultaneously recorded cells 4–7. This increase is possible because each recorded neuron produces extracellular action potential waveforms ('spikes') with a characteristic spatio ­ temporal profile across the recording sites 8–10. The process of using these waveforms to decipher the firing times of the recorded neurons is known as spike sorting 11,12.""",

"""Spike sorting, as currently applied in nearly all labs using extracellular recordings, involves a manual operator. While some labs use a fully manual system, lower error rates can be achieved with a semiautomated process 8, consisting of four steps. First, spikes are detected, typically by high ­pass filtering and thresholding. Second, each spike waveform is summarized by a compact 'feature vector' , typically by principal component analysis. Third, these vectors are divided into groups corresponding to putative neurons using cluster analysis. Finally, the results are manually curated to adjust any errors made by automated algorithms 13. This last step is necessary because although fully automatic spike sorting would be a powerful tool, the output of existing algorithms cannot be accepted without human verification. A similar situation arises in many fields of data ­intensive science: in electron microscopic connectomics, for example, automated methods can only be used under the supervision of human operators 14.""",

"""For tetrode data, this semiautomatic process performs well, reaching error rates of 5% or lower as assessed by ground truth data obtained with simultaneous intracellular recording 8. However, spike sorting methods developed for tetrodes do not work for a newer generation of larger electrode arrays 15,16. This failure occurs for two reasons. First, the automated component can fail in high dimensions; for example, because of the 'curse of dimensionality' that affects cluster analysis in high ­dimensional spaces 17. Second and perhaps more critically, the process of manual curation, while manageable with low ­count probes, cannot scale to the high ­count case without software that guides the operator to only those decisions that cannot be made reliably by a computer. While many different methods for spike sorting have been proposed (for example, refs. 18–24), no method has yet solved these problems robustly enough to be widely adopted by the experimental community.""",

"""Here we describe a system for the spike sorting of high ­channel count electrode data, implemented in a suite of freely available software. While the spike sorting problem has attracted considerable theoretical research, our goal was to produce a practical system that can be immediately used by working neurophysiologists. The ability to process large data sets (millions of spikes in hundreds of dimensions) in reasonable human and computer time was deemed essential; error rates comparable to those of commonly used tetrode methods were deemed acceptable. We tested the software on data recorded from rat neocortex with 32 ­site shank electrodes, as well as data from other species and brain regions. While traditional methods performed extremely poorly on this data, the new algorithms gave close to theoretically optimal performance. The techniques and software have been developed in a community ­led manner, through extensive feedback from a user base of over 320 scientists in 50 neurophysiology labs. The software is downloadable and documented at http://cortexlab.net/tools/
 and is supported by an active user ­group mailing list, klustaviewas@groups.google.com
 .""",

"""RESULTS Our spike sorting pipeline involves three steps: (1) spike detection and feature extraction, (2) cluster analysis, and (3) manual curation. We describe these steps in order. Spike detection The first step of the pipeline is spike detection and feature extraction, implemented by the program SpikeDetekt.""",

"""The primary difference between spike detection for high ­count silicon probes and for tetrodes is that temporally overlapping spikes are extremely common in the former. The spikes seen in these data are diverse ( Fig. 1), with some detected on only one or two channels and others spanning large numbers of channels, as expected of pyramidal cells whose apical dendrites are aligned parallel to the shank 25. In these data, simultaneous firing of multiple neurons is common. However, simultaneously firing neurons are usually detected on distinct sets of channels.""",

"""To deal with the problem of temporally overlapping spikes, we therefore sought to detect spikes as local spatiotemporal events ( Fig. 2 ). This step requires knowledge of the probe geometry, which is specified by the user in the form of an adjacency graph ( Fig. 2 a). We illustrate the spike detection process with reference to a small segment of data containing two temporally overlapping but spatially separated spikes.""",

"""The first stage of the algorithm is high-pass filtering the raw data to remove the slow local field potential signal (Butterworth in forward-backward mode; Fig. 2c). Next, spikes are detected using a double-threshold flood fill algorithm ( Fig. 2 d,e). Specifically, spikes are detected as spatiotemporally connected components, in which the filtered signal exceeds a weak threshold θw for every point and in which at least one point exceeds a strong threshold θs. Optimal values for these parameters were found to be 4 and 2 times the s.d. of the filtered signal, as described below.""",

"""Two points are considered neighboring if they are on a single channel and separated by one time sample, or at a single time point on channels joined by the adjacency graph; this allows the algorithm to work with probes of any geometry, not just linear ones. The dual-threshold approach avoids spurious detection of small noise events because isolated islands in which only the weak threshold is exceeded are not retained. Conversely, spikes will not be erroneously split as a result of noise, as areas joined by weak threshold crossings are merged.""",

"""After detection, spikes are temporally realigned to subsample resolution, to the center of mass of the spike's suprathreshold components, weighted by a power parameter p (see Online Methods). Visual inspection showed that spike times detected with this method corresponded closely to those that would be assigned by a human operator ( Fig. 2 e). The waveforms of each spike are summarized by two vectors.""",

"""First, a feature vector is found by principal component analysis of the realigned waveforms on each channel (three principal components were kept in the analyses reported here). All channels are used in computing the feature vector; thus our two example spikes have similar feature vectors, as their central times are similar ( Fig. 2 f). Second, a mask vector is computed from the peak spike amplitude on each detected channel, rescaled and clipped so channels outside the connected component have mask 0 and channels with amplitude above θs have mask 1. The mask vector allows temporally overlapping spikes to be clustered as coming from separate cells. Indeed, although the feature vectors of our two example spikes were very similar, their mask vectors are completely different ( Fig. 2 g).""",

"""Performance validation and parameter optimization To quantify the performance and optimize the parameters of this algorithm requires 'ground truth': knowledge of when the recorded neurons actually fired. We created a simulated ground truth data set by repeatedly adding the spikes of a 'donor cell' identified in one recording to a second 'acceptor' recording made with same probe. Because the extracellular medium is a linear conductor 26, addition of spike waveforms serves as a sufficient model for overlapping spikes.""",

"""To evaluate the performance of the system, we chose ten donor cells with a variety of amplitudes and waveform distributions ( Fig. 3 a), using recordings from rat cortex with a 32-channel probe shank. To model the variability of waveforms produced by a single neuron due to phenomena such as bursting 27–29, we scaled each spike to a random amplitude in a range that varied by a factor of two (see Online Methods). We refer to the spikes added to the acceptor data set as hybrid spikes and the result as a hybrid data set.""",

"""To evaluate spike detection performance, we used a heuristic criterion to identify which spikes detected by the algorithm corresponded to which hybrid spikes (see Online Methods). We measured performance as a function of three algorithm parameters (θw, θs and p), using four performance statistics. The first statistic was the fraction of hybrid spikes detected ( Fig. 3 b).""",
]


class LiveReaderWindow:
    """
    Live reading window with AI-powered confusion prediction.
    
    Instead of manual labeling, this window:
    - Shows text for reading
    - Tracks cursor position (which word is being read)
    - Makes real-time predictions using trained RNN
    - Highlights predicted confusing words when 'H' is pressed
    """
    
    def __init__(self, parent_visualizer, model_path: str = None):
        self.parent = parent_visualizer
        self.root = tk.Tk()
        self.root.title("🧠 NEUROADAPTIVE READER - Live Confusion Detection")
        
        # Window setup
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        self.active = True
        
        # Prediction mode (instead of labeling)
        self.prediction_mode = True
        self.show_predictions = False  # Toggle with 'H'
        
        # Cursor tracking
        self.current_word = ""
        self.current_word_start = None
        self.current_word_end = None
        
        # Pre-calculated word positions for FAST lookup
        self.word_positions = []
        self.current_word_index = -1
        
        # Word instance tracking
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        
        # Return sweep detection
        self.last_cursor_x = 0
        self.in_return_sweep = False
        self.return_sweep_threshold = 150
        
        # Prediction tracking per word position
        # Key: (text_idx, char_start), Value: list of prediction probabilities
        self.word_predictions = {}
        
        # Currently highlighted predictions
        self.prediction_highlights = []
        
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
                text="🧠 NEUROADAPTIVE READER",
                font=('Arial', 24, 'bold'),
                fg='#00FF88',
                bg='#1a1a1a').pack(pady=10)
        
        tk.Label(header_frame,
                text="H: Toggle Prediction Highlights | Space: Start/Stop | ←/→: Change Text",
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
        
        # Configure highlight colors
        self.text_display.tag_configure("current_word", underline=True, foreground="#FFD93D")
        self.text_display.tag_configure("predicted_confused", background="#FF4444", foreground="white")
        self.text_display.tag_configure("predicted_confused_high", background="#FF0000", foreground="white")
        self.text_display.tag_configure("predicted_confused_low", background="#884444", foreground="white")
        
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
                                        text="⏸ NOT RECORDING",
                                        font=('Arial', 16, 'bold'),
                                        fg='#888888',
                                        bg='#1a1a1a')
        self.recording_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.prediction_status = tk.Label(status_frame,
                                         text="🤖 AI READY",
                                         font=('Arial', 16, 'bold'),
                                         fg='#00FF88',
                                         bg='#1a1a1a')
        self.prediction_status.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.confusion_meter = tk.Label(status_frame,
                                       text="Confusion: --",
                                       font=('Arial', 16),
                                       fg='#FFD93D',
                                       bg='#1a1a1a')
        self.confusion_meter.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Word status
        self.word_status = tk.Label(status_frame,
                                   text="Current word: -",
                                   font=('Arial', 14, 'italic'),
                                   fg='#FFD93D',
                                   bg='#1a1a1a')
        self.word_status.pack(side=tk.RIGHT, padx=20, pady=5)
        
        tk.Label(status_frame,
                text="↑/↓: Scroll | H: Show/Hide Predictions | +/-: Font",
                font=('Arial', 12),
                fg='#888888',
                bg='#1a1a1a').pack(side=tk.BOTTOM, padx=20, pady=5)
        
        self.prediction_label = tk.Label(status_frame,
                                        text="Predictions: Hidden (press H)",
                                        font=('Arial', 12),
                                        fg='#FF6B6B',
                                        bg='#1a1a1a')
        self.prediction_label.pack(side=tk.BOTTOM, padx=20, pady=2)
    
    def _bind_events(self):
        """Bind all event handlers"""
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Mouse events
        self.text_display.bind('<Motion>', self.on_mouse_motion)
        self.text_display.bind('<Leave>', self.on_mouse_leave)
    
    def _build_word_index(self):
        """Pre-calculate word positions for FAST cursor tracking."""
        self.word_positions = []
        text = TRAINING_TEXTS[self.parent.current_text_index]
        
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            
            word = text[start:i]
            tk_start = f"1.{start}"
            tk_end = f"1.{i}"
            self.word_positions.append((start, i, word, tk_start, tk_end))
        
        self.current_word_index = -1
    
    def _find_word_at_char(self, char_pos):
        """Binary search to find word index at character position."""
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
    
    def on_mouse_motion(self, event):
        """Track cursor position with prediction updates."""
        cursor_x = event.x
        
        # Return sweep detection
        if cursor_x < self.last_cursor_x - self.return_sweep_threshold:
            self.in_return_sweep = True
        
        if self.in_return_sweep and cursor_x > self.last_cursor_x:
            self.in_return_sweep = False
        
        self.last_cursor_x = cursor_x
        
        if self.in_return_sweep:
            return
        
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            col = int(index.split('.')[1])
            new_word_idx = self._find_word_at_char(col)
            
            if new_word_idx >= 0 and new_word_idx != self.current_word_index:
                char_start, char_end, word, tk_start, tk_end = self.word_positions[new_word_idx]
                
                self.current_word = word
                self.current_word_char_start = char_start
                self.current_word_char_end = char_end
                self.parent.current_word = word
                self.parent.current_word_char_start = char_start
                self.parent.current_word_char_end = char_end
                
                # Remove old underline
                if self.current_word_start and self.current_word_end:
                    self.text_display.tag_remove("current_word",
                                                  self.current_word_start,
                                                  self.current_word_end)
                
                self.text_display.tag_add("current_word", tk_start, tk_end)
                self.current_word_start = tk_start
                self.current_word_end = tk_end
                self.current_word_index = new_word_idx
                
                self.word_status.config(text=f"Current word: {word}")
                
                # Update prediction for this word position
                self._update_word_prediction()
        except:
            pass
    
    def _update_word_prediction(self):
        """Get prediction for current word position."""
        if not self.parent.is_recording:
            return
        
        # Get current prediction from model
        prediction = self.parent.get_current_prediction()
        if prediction >= 0:
            key = (self.parent.current_text_index, self.current_word_char_start)
            if key not in self.word_predictions:
                self.word_predictions[key] = []
            self.word_predictions[key].append(prediction)
            
            # Update confusion meter
            self.confusion_meter.config(
                text=f"Confusion: {prediction*100:.0f}%",
                fg=self._get_confusion_color(prediction)
            )
    
    def _get_confusion_color(self, prob):
        """Get color based on confusion probability."""
        if prob < 0.3:
            return '#00FF88'  # Green - not confused
        elif prob < 0.5:
            return '#FFD93D'  # Yellow - uncertain
        elif prob < 0.7:
            return '#FF8844'  # Orange - probably confused
        else:
            return '#FF4444'  # Red - confused
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving text area"""
        self.current_word = ""
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        self.parent.current_word = ""
        self.parent.current_word_char_start = -1
        self.parent.current_word_char_end = -1
        self.word_status.config(text="Current word: -")
        
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
            'Left': lambda: self.change_text(-1),
            'Right': lambda: self.change_text(1),
            'space': lambda: (self.parent.toggle_recording(), self.update_status()),
        }
        
        if event.keysym in key_actions:
            key_actions[event.keysym]()
        elif event.char.lower() == 'q':
            self.on_close()
        elif event.char.lower() == 'h':
            self.toggle_predictions()
        elif event.char in ['+', '=']:
            self.adjust_font_size(2)
        elif event.char == '-':
            self.adjust_font_size(-2)
    
    def change_text(self, delta):
        """Change to next/previous text"""
        if delta > 0:
            self.parent.next_text()
        else:
            self.parent.previous_text()
        
        # Clear predictions for old text
        self.word_predictions = {
            k: v for k, v in self.word_predictions.items() 
            if k[0] == self.parent.current_text_index
        }
        
        self.update_display()
    
    def toggle_predictions(self):
        """Toggle display of prediction highlights"""
        self.show_predictions = not self.show_predictions
        
        if self.show_predictions:
            self._show_prediction_highlights()
            self.prediction_label.config(
                text="Predictions: VISIBLE (press H to hide)",
                fg='#FF4444'
            )
        else:
            self._clear_prediction_highlights()
            self.prediction_label.config(
                text="Predictions: Hidden (press H)",
                fg='#888888'
            )
    
    def _show_prediction_highlights(self):
        """Highlight words predicted as confusing"""
        self._clear_prediction_highlights()
        
        current_text = self.parent.current_text_index
        threshold = 0.5
        min_samples = 2
        
        confused_words = []
        for (text_idx, char_start), predictions in self.word_predictions.items():
            if text_idx == current_text and len(predictions) >= min_samples:
                avg_prob = np.mean(predictions)
                if avg_prob >= threshold:
                    confused_words.append((char_start, avg_prob))
        
        if not confused_words:
            messagebox.showinfo(
                "No Predictions",
                "No words have been predicted as confusing yet.\n\n"
                "Make sure you're recording and reading the text."
            )
            return
        
        # Highlight each confused word
        for char_start, prob in confused_words:
            # Find word ending
            word_idx = self._find_word_at_char(char_start)
            if word_idx >= 0:
                _, char_end, word, tk_start, tk_end = self.word_positions[word_idx]
                
                # Choose highlight intensity based on probability
                if prob >= 0.8:
                    tag = "predicted_confused_high"
                elif prob >= 0.6:
                    tag = "predicted_confused"
                else:
                    tag = "predicted_confused_low"
                
                self.text_display.tag_add(tag, tk_start, tk_end)
                self.prediction_highlights.append((tk_start, tk_end, tag))
        
        n_confused = len(confused_words)
        self.prediction_label.config(
            text=f"Predictions: {n_confused} words highlighted",
            fg='#FF4444'
        )
    
    def _clear_prediction_highlights(self):
        """Remove all prediction highlights"""
        for tk_start, tk_end, tag in self.prediction_highlights:
            self.text_display.tag_remove(tag, tk_start, tk_end)
        self.prediction_highlights = []
    
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
        
        # Reset tracking
        self.current_word = ""
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        self.parent.current_word = ""
        self.parent.current_word_char_start = -1
        self.parent.current_word_char_end = -1
        self.word_status.config(text="Current word: -")
        self.current_word_start = None
        self.current_word_end = None
        self.current_word_index = -1
        self._build_word_index()
        
        # Clear old highlights
        self._clear_prediction_highlights()
        self.show_predictions = False
        self.prediction_label.config(
            text="Predictions: Hidden (press H)",
            fg='#888888'
        )
    
    def update_status(self):
        """Update recording status"""
        if self.parent.is_recording:
            elapsed = time.time() - self.parent.recording_start_time
            status_text = f"⏺ RECORDING: {elapsed:.1f}s"
            self.recording_status.config(text=status_text, fg='#ff4444')
            
            # Update AI status
            if self.parent.predictor and self.parent.predictor.preprocessor.n_samples_seen > 0:
                warmup_pct = min(100, 100 * self.parent.predictor.preprocessor.n_samples_seen / 
                               self.parent.predictor.preprocessor.warmup_samples)
                if warmup_pct < 100:
                    self.prediction_status.config(
                        text=f"🤖 Warming up... {warmup_pct:.0f}%",
                        fg='#FFD93D'
                    )
                else:
                    self.prediction_status.config(
                        text="🤖 AI PREDICTING",
                        fg='#00FF88'
                    )
        else:
            self.recording_status.config(text="⏸ NOT RECORDING", fg='#888888')
            self.confusion_meter.config(text="Confusion: --", fg='#FFD93D')
    
    def track_cursor(self):
        """Low-frequency cursor polling as backup"""
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


class LiveEEGVisualizer:
    """
    EEG visualizer with live confusion prediction.
    
    Receives data from Muse S headset and runs RNN inference in real-time.
    """
    
    def __init__(self, port=8052, buffer_size=2000, window_duration=10, model_path=None):
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
        
        # Performance optimization
        self.info_text_objects = {}
        self.info_panel_initialized = False
        self.spectral_update_counter = 0
        self.spectral_update_interval = 4
        self.plot_window_focused = False
        
        # UI elements
        self.reader_window = None
        self.current_text_index = 0
        self.current_word = ""
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        
        # Recording (now just for data collection, not labeling)
        self.is_recording = False
        self.recording_start_time = None
        self.record_button = None
        
        # EEG/fNIRS buffers for prediction
        self.prediction_eeg_buffer = deque(maxlen=512)  # 2 seconds
        self.prediction_fnirs_buffer = deque(maxlen=512)
        
        # Last values for interpolation
        self.last_eeg_data = None
        self.last_fnirs_data = None
        self.last_motion_data = None
        self.last_ref_data = None
        
        # Stats
        self.packet_count = 0
        self.eeg_packet_count = 0
        self.fnirs_packet_count = 0
        
        # Confusion predictor
        self.predictor = None
        self.model_path = model_path
        self._load_predictor()
        
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
    
    def _load_predictor(self):
        """Load the trained confusion prediction model."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                print(f"\n🤖 Loading confusion detection model: {self.model_path}")
                self.predictor = LiveConfusionPredictor(
                    model_path=self.model_path,
                    device='cpu',
                    prediction_threshold=0.5,
                    smoothing_window=4
                )
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.predictor = None
        else:
            print("\n⚠️ No model path provided or model not found.")
            print("   Run train_confusion_rnn.py first to train a model.")
            self.predictor = None
    
    def get_current_prediction(self) -> float:
        """Get current confusion prediction from the model."""
        if self.predictor is None:
            return -1
        
        if len(self.prediction_eeg_buffer) < 256:  # Need at least 1 second
            return -1
        
        # Get recent data
        eeg = np.array(list(self.prediction_eeg_buffer))
        fnirs = np.array(list(self.prediction_fnirs_buffer))
        
        # Get prediction
        return self.predictor.predict(eeg, fnirs)
    
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
                    
                    # Add to prediction buffer
                    if self.is_recording:
                        self.prediction_eeg_buffer.append(args)
                        fnirs = self.last_fnirs_data if self.last_fnirs_data else [0] * 8
                        self.prediction_fnirs_buffer.append(fnirs)
                    
                    if self.eeg_packet_count <= 3:
                        print(f"EEG packet {self.eeg_packet_count}: {args}")
                
                elif data_type == 'optics' and len(args) == 8:
                    for i, ch in enumerate([f'Ch{j}_{t}' for j in range(1,5) for t in ['norm', 'raw']]):
                        self.fnirs_channels[ch].append(args[i])
                    self.fnirs_packet_count += 1
                    self.last_fnirs_data = args
                    
                    if self.fnirs_packet_count <= 3:
                        print(f"fNIRS packet {self.fnirs_packet_count}: norm={args[:4]}")
                
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
    
    def toggle_recording(self, event=None):
        """Toggle recording state"""
        if not self.is_recording:
            with self.lock:
                self.is_recording = True
                self.recording_start_time = time.time()
                self.prediction_eeg_buffer.clear()
                self.prediction_fnirs_buffer.clear()
            
            if self.predictor:
                self.predictor.clear_predictions()
            
            if self.record_button:
                self.record_button.label.set_text('Stop Recording')
                self.record_button.color = '#ff4444'
            
            print(f"\n{'='*50}")
            print(f"RECORDING STARTED - AI will predict confusion in real-time")
            print(f"{'='*50}")
        else:
            self.is_recording = False
            
            if self.record_button:
                self.record_button.label.set_text('Start Recording')
                self.record_button.color = '#2a2a2a'
            
            print(f"\n{'='*50}")
            print(f"RECORDING STOPPED - Press H in reader to see predictions")
            print(f"{'='*50}")
    
    def cleanup_on_exit(self):
        """Clean exit handler"""
        if self.shutting_down:
            return
        self.shutting_down = True
        self.running = False
        if self.socket:
            self.socket.close()
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n\nReceived interrupt.")
        self.cleanup_on_exit()
        sys.exit(0)
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        gs = GridSpec(6, 2, figure=self.fig, 
                     height_ratios=[3, 3, 3, 2, 2, 1],
                     width_ratios=[4, 1],
                     hspace=0.3)
        
        self.axes = {
            'eeg': self.fig.add_subplot(gs[0, 0]),
            'spectral': self.fig.add_subplot(gs[1, 0]),
            'fnirs': self.fig.add_subplot(gs[2, 0]),
            'motion': self.fig.add_subplot(gs[3, 0]),
            'gyro': self.fig.add_subplot(gs[4, 0]),
            'ref': self.fig.add_subplot(gs[5, 0]),
            'info': self.fig.add_subplot(gs[:5, 1])
        }
        
        for name, ax in self.axes.items():
            ax.set_facecolor('#1a1a1a')
            if name != 'info':
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
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
        
        self.axes['eeg'].set_ylabel('Amplitude (µV)')
        self.axes['spectral'].set_ylabel('Power (dB)')
        self.axes['spectral'].set_xlabel('Frequency (Hz)')
        self.axes['fnirs'].set_ylabel('Intensity')
        self.axes['motion'].set_ylabel('Acceleration (g)')
        self.axes['gyro'].set_ylabel('Angular velocity (°/s)')
        self.axes['ref'].set_ylabel('Voltage')
        self.axes['ref'].set_xlabel('Time (s)')
        
        self.axes['info'].set_xticks([])
        self.axes['info'].set_yticks([])
        for spine in self.axes['info'].spines.values():
            spine.set_visible(False)
        
        ax_button = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.record_button = Button(ax_button, 'Start Recording', 
                                   color='#2a2a2a', hovercolor='#3a3a3a')
        self.record_button.on_clicked(self.toggle_recording)
        
        self._initialize_lines()
        plt.tight_layout()
    
    def _initialize_lines(self):
        """Initialize plot lines"""
        for ch, color in self.eeg_colors.items():
            line, = self.axes['eeg'].plot([], [], label=ch, color=color, 
                                         linewidth=1.5, alpha=0.95)
            self.lines[f'eeg_{ch}'] = line
        
        self.spectral_lines = {}
        for ch, color in self.eeg_colors.items():
            line, = self.axes['spectral'].plot([], [], label=ch, color=color, 
                                              linewidth=1.5, alpha=0.9)
            self.spectral_lines[ch] = line
        
        self.axes['spectral'].set_xlim(0, self.max_freq)
        
        for band_name, (low, high) in self.freq_bands.items():
            self.axes['spectral'].axvline(x=low, color='white', linestyle=':', 
                                         alpha=0.3, linewidth=0.5)
            mid_freq = (low + high) / 2
            if mid_freq < self.max_freq:
                self.axes['spectral'].text(mid_freq, 0.98, band_name, 
                                         fontsize=8, color='white', alpha=0.7,
                                         ha='center', va='top',
                                         transform=self.axes['spectral'].get_xaxis_transform())
        
        for i, (ch_base, color) in enumerate(self.fnirs_colors.items()):
            ch_norm = f'{ch_base}_norm'
            line, = self.axes['fnirs'].plot([], [], label=ch_base, 
                                          color=color, linewidth=1.5, alpha=0.9)
            self.lines[f'fnirs_{ch_norm}'] = line
        
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
        
        ref_colors = ['#95A5A6', '#7F8C8D']
        for i, ch in enumerate(['DRL', 'REF']):
            line, = self.axes['ref'].plot([], [], label=ch,
                                        color=ref_colors[i],
                                        linewidth=1.5, alpha=0.9)
            self.lines[f'ref_{ch}'] = line
        
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
        if not self.plot_window_focused:
            return list(self.lines.values()) + list(self.spectral_lines.values())
        
        with self.lock:
            if len(self.timestamps) < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            min_eeg_length = min(len(self.eeg_channels[ch]) for ch in self.eeg_channels 
                               if len(self.eeg_channels[ch]) > 0)
            
            if min_eeg_length < 2:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            timestamps = np.array(list(self.timestamps)[-min_eeg_length:])
            if len(timestamps) > 1:
                time_axis = timestamps - timestamps[-1]
                display_mask = time_axis >= -self.window_duration
            else:
                return list(self.lines.values()) + list(self.spectral_lines.values())
            
            filtered_eeg_data = {}
            eeg_values_for_scaling = []
            
            for ch_name in self.eeg_channels:
                if ch_name in self.eeg_channels and len(self.eeg_channels[ch_name]) >= min_eeg_length:
                    line_key = f'eeg_{ch_name}'
                    if line_key in self.lines:
                        data_array = np.array(list(self.eeg_channels[ch_name])[-min_eeg_length:])
                        
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
                        
                        display_time = time_axis[display_mask]
                        display_data = filtered_data[display_mask]
                        
                        self.lines[line_key].set_data(display_time, display_data)
                        eeg_values_for_scaling.extend(display_data)
            
            self.spectral_update_counter += 1
            if self.spectral_update_counter >= self.spectral_update_interval and len(filtered_eeg_data) == 4:
                self.spectral_update_counter = 0
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
            
            for channel_dict, prefix in [(self.fnirs_channels, 'fnirs'), 
                                        (self.motion_channels, 'motion'),
                                        (self.ref_channels, 'ref')]:
                for ch_name, data_deque in channel_dict.items():
                    if len(data_deque) > 0:
                        if prefix == 'fnirs' and ch_name.endswith('_norm'):
                            line_key = f'{prefix}_{ch_name}'
                        elif prefix == 'fnirs' and ch_name.endswith('_raw'):
                            continue
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
            
            for ax in [self.axes['eeg'], self.axes['fnirs'], 
                      self.axes['motion'], self.axes['gyro'], self.axes['ref']]:
                ax.set_xlim(-self.window_duration, 0)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            
            if eeg_values_for_scaling:
                eeg_std = np.std(eeg_values_for_scaling)
                eeg_median = np.median(eeg_values_for_scaling)
                y_range = 4 * eeg_std
                self.axes['eeg'].set_ylim(eeg_median - y_range/2, eeg_median + y_range/2)
            
            self._update_info_panel()
        
        return list(self.lines.values()) + list(self.spectral_lines.values())
    
    def _init_info_panel(self):
        """Initialize info panel"""
        ax = self.axes['info']
        t = self.info_text_objects
        y = 0.95
        
        ax.text(0.1, y, '🧠 Live Confusion Detection', fontsize=12, weight='bold', 
               color='#00FF88', transform=ax.transAxes)
        
        y -= 0.06
        t['rec_status'] = ax.text(0.1, y, '', fontsize=10, color='#ff4444', 
                                  weight='bold', transform=ax.transAxes)
        y -= 0.04
        t['model_status'] = ax.text(0.1, y, '', fontsize=9, color='#00FF88', transform=ax.transAxes)
        y -= 0.04
        t['prediction'] = ax.text(0.1, y, '', fontsize=9, color='#FFD93D', transform=ax.transAxes)
        
        y -= 0.06
        t['word'] = ax.text(0.1, y, 'Word: -', fontsize=9, color='#FFD93D', transform=ax.transAxes)
        y -= 0.04
        t['text_idx'] = ax.text(0.1, y, f'Text: 1/{len(TRAINING_TEXTS)}', fontsize=9, 
                               color='#aaaaaa', transform=ax.transAxes)
        
        y -= 0.08
        ax.text(0.1, y, 'EEG (4ch):', fontsize=10, weight='bold', color='#4ECDC4', transform=ax.transAxes)
        y -= 0.05
        
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            ax.text(0.15, y, f'{ch}:', fontsize=9, color=self.eeg_colors[ch], transform=ax.transAxes)
            t[f'eeg_{ch}'] = ax.text(0.4, y, '0.0±0.0', fontsize=9, color='white', transform=ax.transAxes)
            y -= 0.04
        
        y -= 0.06
        ax.text(0.1, y, 'System:', fontsize=10, weight='bold', color='#95A5A6', transform=ax.transAxes)
        y -= 0.05
        
        for label, color in [('Packets:', 'white'), ('EEG:', '#4ECDC4'), ('fNIRS:', '#E74C3C')]:
            ax.text(0.15, y, label, fontsize=9, color='white', transform=ax.transAxes)
            t[label] = ax.text(0.4, y, '0', fontsize=9, color=color, transform=ax.transAxes)
            y -= 0.04
        
        self.info_panel_initialized = True
    
    def _update_info_panel(self):
        """Update info panel"""
        if not self.info_panel_initialized:
            self._init_info_panel()
        
        t = self.info_text_objects
        
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            t['rec_status'].set_text(f'⏺ REC: {elapsed:.1f}s')
        else:
            t['rec_status'].set_text('')
        
        if self.predictor:
            t['model_status'].set_text('🤖 Model: Active')
            pred = self.get_current_prediction()
            if pred >= 0:
                t['prediction'].set_text(f'Confusion: {pred*100:.0f}%')
            else:
                t['prediction'].set_text('Confusion: --')
        else:
            t['model_status'].set_text('⚠️ No model loaded')
            t['prediction'].set_text('')
        
        t['word'].set_text(f'Word: {self.current_word[:15] if self.current_word else "-"}')
        t['text_idx'].set_text(f'Text: {self.current_text_index + 1}/{len(TRAINING_TEXTS)}')
        
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if len(self.eeg_channels[ch]) > 0:
                data = np.array(list(self.eeg_channels[ch])[-100:])
                t[f'eeg_{ch}'].set_text(f'{np.mean(data):.1f}±{np.std(data):.1f}')
        
        t['Packets:'].set_text(str(self.packet_count))
        t['EEG:'].set_text(str(self.eeg_packet_count))
        t['fNIRS:'].set_text(str(self.fnirs_packet_count))
    
    def start(self):
        """Start the visualizer"""
        print("\n" + "="*60)
        print("   🧠 NEUROADAPTIVE READER - LIVE CONFUSION DETECTION")
        print("="*60)
        print(f"\n📡 Listening for OSC data on UDP port {self.port}")
        print("\n🤖 LIVE PREDICTION MODE:")
        print("  • The AI predicts confusion as you read")
        print("  • Press H to see prediction highlights")
        print("  • No manual labeling required!")
        
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
        
        self.setup_visualization()
        
        print("\n🖥️ Opening reader window...")
        self.reader_window = LiveReaderWindow(self, self.model_path)
        self.reader_window.update_loop()
        
        def on_key(event):
            if event.key == 'q':
                print("\nQuitting...")
                if self.reader_window and self.reader_window.active:
                    self.reader_window.on_close()
                self.cleanup_on_exit()
                plt.close('all')
                self.stop()
            elif event.key in ['+', '=']:
                self.window_duration = min(self.window_duration + 2, 30)
            elif event.key == '-':
                self.window_duration = max(self.window_duration - 2, 2)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        def on_plot_focus_in(event):
            self.plot_window_focused = True
        
        def on_plot_focus_out(event):
            self.plot_window_focused = False
        
        try:
            plot_window = self.fig.canvas.manager.window
            plot_window.bind('<FocusIn>', on_plot_focus_in)
            plot_window.bind('<FocusOut>', on_plot_focus_out)
            plot_window.bind('<Map>', on_plot_focus_in)
            plot_window.bind('<Unmap>', on_plot_focus_out)
        except:
            self.plot_window_focused = True
        
        def on_close(event):
            if self.reader_window and self.reader_window.active:
                self.reader_window.on_close()
            self.cleanup_on_exit()
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=80,
            blit=False,
            cache_frame_data=False
        )
        
        print("\n✅ Live reader started!")
        print("🖥️ Focus the reader window to begin")
        print("\n" + "="*60)
        
        try:
            def update_matplotlib():
                try:
                    if self.fig and plt.fignum_exists(self.fig.number):
                        if self.plot_window_focused:
                            self.fig.canvas.draw_idle()
                            self.fig.canvas.flush_events()
                except:
                    pass
                if self.reader_window and self.reader_window.active:
                    self.reader_window.root.after(80, update_matplotlib)
            
            plt.show(block=False)
            update_matplotlib()
            self.reader_window.root.mainloop()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            if self.reader_window and self.reader_window.active:
                self.reader_window.on_close()
            self.cleanup_on_exit()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the visualizer"""
        self.running = False
        if self.socket:
            self.socket.close()
        
        if self.reader_window and self.reader_window.active:
            self.reader_window.on_close()
        
        print(f"\n{'='*50}")
        print(f"LIVE READER STOPPED")
        print(f"Total packets: {self.packet_count}")
        print(f"EEG: {self.eeg_packet_count}, fNIRS: {self.fnirs_packet_count}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Live neuroadaptive reader with confusion detection')
    
    parser.add_argument('--model', type=str, default='confusion_rnn_best.pth',
                       help='Path to trained model file')
    parser.add_argument('--port', type=int, default=8052,
                       help='UDP port for OSC data')
    
    args = parser.parse_args()
    
    visualizer = LiveEEGVisualizer(
        port=args.port, 
        buffer_size=2000, 
        window_duration=10,
        model_path=args.model
    )
    
    try:
        visualizer.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
