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
import re
import hashlib


# =============================================================================
# WORD EMBEDDING SYSTEM
# =============================================================================
# Provides semantic vectors for words to help RNN understand word difficulty/type
# Multiple fallback options: sentence-transformers > GloVe > character hash

class WordEmbeddings:
    """
    Word embedding system with multiple fallback options.
    
    Priority order:
    1. sentence-transformers (best: contextual, handles OOV, phrases)
    2. GloVe (good: pre-trained, fast lookup, needs download)
    3. Character hash (fallback: deterministic but no semantics)
    """
    
    def __init__(self, embedding_dim=384, glove_path=None):
        self.embedding_dim = embedding_dim
        self.embeddings = {}  # word -> vector cache
        self.method = None
        self.model = None
        
        # Try loading in priority order
        if self._try_sentence_transformers():
            print("✅ Word embeddings: Using sentence-transformers (best quality)")
        elif self._try_glove(glove_path):
            print(f"✅ Word embeddings: Using GloVe ({len(self.embeddings)} words)")
        else:
            self._use_character_hash()
            print("⚠️ Word embeddings: Using character hash fallback")
            print("   For better results, install: pip3 install sentence-transformers")
    
    def _try_sentence_transformers(self) -> bool:
        """Try loading sentence-transformers (best option)."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model optimized for semantic similarity
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  # This model outputs 384-dim vectors
            self.method = 'sentence_transformers'
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"   sentence-transformers error: {e}")
            return False
    
    def _try_glove(self, glove_path=None) -> bool:
        """Try loading GloVe embeddings."""
        # Common paths to check
        search_paths = [
            glove_path,
            os.path.expanduser("~/Documents/EEGAssistant/glove.6B.300d.txt"),
            os.path.expanduser("~/glove.6B.300d.txt"),
            os.path.expanduser("~/Downloads/glove.6B.300d.txt"),
            "glove.6B.300d.txt",
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                try:
                    print(f"   Loading GloVe from {path}...")
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            values = line.strip().split()
                            if len(values) > 2:
                                word = values[0]
                                vector = np.array(values[1:], dtype=np.float32)
                                self.embeddings[word] = vector
                    
                    if self.embeddings:
                        # Get dimension from first vector
                        self.embedding_dim = len(next(iter(self.embeddings.values())))
                        self.method = 'glove'
                        return True
                except Exception as e:
                    print(f"   GloVe load error: {e}")
        
        return False
    
    def _use_character_hash(self):
        """Fallback: deterministic character-based embeddings."""
        self.embedding_dim = 128  # Smaller for hash-based
        self.method = 'char_hash'
    
    def _char_hash_embed(self, word: str) -> np.ndarray:
        """Generate deterministic embedding from character hashes."""
        # Normalize word
        word_lower = word.lower().strip()
        
        # Create embedding from multiple hash perspectives
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Hash the full word
        h = hashlib.md5(word_lower.encode()).digest()
        for i, b in enumerate(h):
            embedding[i % self.embedding_dim] += (b - 128) / 128.0
        
        # Hash character n-grams (captures morphology)
        for n in [2, 3, 4]:
            for i in range(len(word_lower) - n + 1):
                ngram = word_lower[i:i+n]
                h = hashlib.md5(ngram.encode()).digest()
                offset = (n - 2) * 16 + 48  # Different region for each n
                for j, b in enumerate(h):
                    embedding[(offset + j) % self.embedding_dim] += (b - 128) / 256.0
        
        # Add word length feature
        embedding[0] = len(word_lower) / 20.0  # Normalize by typical max
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word (cached)."""
        # Check cache first
        word_key = word.lower().strip()
        if word_key in self.embeddings:
            return self.embeddings[word_key]
        
        # Compute embedding based on method
        if self.method == 'sentence_transformers':
            # sentence-transformers handles everything
            embedding = self.model.encode(word, convert_to_numpy=True)
        elif self.method == 'glove':
            # GloVe lookup with fallback
            if word_key in self.embeddings:
                return self.embeddings[word_key]
            # Try without punctuation
            clean_word = re.sub(r'[^\w]', '', word_key)
            if clean_word in self.embeddings:
                embedding = self.embeddings[clean_word]
            else:
                # Unknown word: use character hash
                embedding = self._char_hash_embed(word_key)
        else:
            # Character hash fallback
            embedding = self._char_hash_embed(word_key)
        
        # Cache and return
        self.embeddings[word_key] = embedding
        return embedding
    
    def get_embeddings_batch(self, words: list) -> np.ndarray:
        """Get embeddings for multiple words efficiently."""
        if self.method == 'sentence_transformers':
            # Batch encode is much faster
            return self.model.encode(words, convert_to_numpy=True)
        else:
            return np.array([self.get_embedding(w) for w in words])
    
    def precompute_text_embeddings(self, texts: list) -> dict:
        """
        Pre-compute embeddings for all unique words in texts.
        Returns dict mapping (text_idx, char_start) -> embedding
        """
        all_words = set()
        word_positions = {}  # (text_idx, char_start) -> word
        
        for text_idx, text in enumerate(texts):
            i = 0
            while i < len(text):
                # Skip whitespace
                while i < len(text) and text[i].isspace():
                    i += 1
                if i >= len(text):
                    break
                
                # Find word
                start = i
                while i < len(text) and not text[i].isspace():
                    i += 1
                
                word = text[start:i]
                all_words.add(word.lower())
                word_positions[(text_idx, start)] = word
        
        # Batch compute embeddings
        word_list = list(all_words)
        if word_list:
            print(f"   Pre-computing embeddings for {len(word_list)} unique words...")
            embeddings = self.get_embeddings_batch(word_list)
            for word, emb in zip(word_list, embeddings):
                self.embeddings[word.lower()] = emb
        
        # Build position -> embedding map
        position_embeddings = {}
        for (text_idx, char_start), word in word_positions.items():
            position_embeddings[(text_idx, char_start)] = self.get_embedding(word)
        
        return position_embeddings


# Global embedding instance (initialized lazily)
_word_embeddings = None

def get_word_embeddings():
    """Get or create the global word embeddings instance."""
    global _word_embeddings
    if _word_embeddings is None:
        _word_embeddings = WordEmbeddings()
    return _word_embeddings

# Training texts
TRAINING_TEXTS = [
"""Developments in microfabrication technology have enabled the production of neural electrode arrays with hundreds of closely spaced recording sites, and electrodes with thousands of sites are under development. These probes in principle allow the simultaneous recording of very large numbers of neurons. However, use of this technology requires the development of techniques for decoding the spike times of the recorded neurons from the raw data captured from the probes. Here we present a set of tools to solve this problem, implemented in a suite of practical, user-friendly, open-source software. We validate these methods on data from the cortex, hippocampus and thalamus of rat, mouse, macaque and marmoset, demonstrating error rates as low as 5%.""",

"""One of the most powerful techniques for neuronal population recording is extracellular electrophysiology using microfabricated electrode arrays 1–3. Advances in microfabrication have continually increased the number of recording sites available on neural probes, and the number of recordable neurons is further increased by having closely spaced recording sites. Indeed, while a single sharp electrode can provide good isolation of one or two neurons, placing as few as four recording sites together in a tetrode can reveal the firing patterns of 10–20 simultaneously recorded cells 4–7. This increase is possible because each recorded neuron produces extracellular action potential waveforms (‘spikes’) with a characteristic spatio ­ temporal profile across the recording sites 8–10. The process of using these waveforms to decipher the firing times of the recorded neurons is known as spike sorting 11,12.""",

"""Spike sorting, as currently applied in nearly all labs using extracellular recordings, involves a manual operator. While some labs use a fully manual system, lower error rates can be achieved with a semiautomated process 8, consisting of four steps. First, spikes are detected, typically by high ­pass filtering and thresholding. Second, each spike waveform is summarized by a compact ‘feature vector’ , typically by principal component analysis. Third, these vectors are divided into groups corresponding to putative neurons using cluster analysis. Finally, the results are manually curated to adjust any errors made by automated algorithms 13. This last step is necessary because although fully automatic spike sorting would be a powerful tool, the output of existing algorithms cannot be accepted without human verification. A similar situation arises in many fields of data ­intensive science: in electron microscopic connectomics, for example, automated methods can only be used under the supervision of human operators 14.""",

"""For tetrode data, this semiautomatic process performs well, reaching error rates of 5% or lower as assessed by ground truth data obtained with simultaneous intracellular recording 8. However, spike sorting methods developed for tetrodes do not work for a newer generation of larger electrode arrays 15,16. This failure occurs for two reasons. First, the automated component can fail in high dimensions; for example, because of the ‘curse of dimensionality’ that affects cluster analysis in high ­dimensional spaces 17. Second and perhaps more critically, the process of manual curation, while manageable with low ­count probes, cannot scale to the high ­count case without software that guides the operator to only those decisions that cannot be made reliably by a computer. While many different methods for spike sorting have been proposed (for example, refs. 18–24), no method has yet solved these problems robustly enough to be widely adopted by the experimental community.""",

"""Here we describe a system for the spike sorting of high ­channel count electrode data, implemented in a suite of freely available software. While the spike sorting problem has attracted considerable theoretical research, our goal was to produce a practical system that can be immediately used by working neurophysiologists. The ability to process large data sets (millions of spikes in hundreds of dimensions) in reasonable human and computer time was deemed essential; error rates comparable to those of commonly used tetrode methods were deemed acceptable. We tested the software on data recorded from rat neocortex with 32 ­site shank electrodes, as well as data from other species and brain regions. While traditional methods performed extremely poorly on this data, the new algorithms gave close to theoretically optimal performance. The techniques and software have been developed in a community ­led manner, through extensive feedback from a user base of over 320 scientists in 50 neurophysiology labs. The software is downloadable and documented at http://cortexlab.net/tools/
 and is supported by an active user ­group mailing list, klustaviewas@groups.google.com
 .""",

"""RESULTS Our spike sorting pipeline involves three steps: (1) spike detection and feature extraction, (2) cluster analysis, and (3) manual curation. We describe these steps in order. Spike detection The first step of the pipeline is spike detection and feature extraction, implemented by the program SpikeDetekt.""",

"""The primary difference between spike detection for high ­count silicon probes and for tetrodes is that temporally overlapping spikes are extremely common in the former. The spikes seen in these data are diverse ( Fig. 1), with some detected on only one or two channels and others spanning large numbers of channels, as expected of pyramidal cells whose apical dendrites are aligned parallel to the shank 25. In these data, simultaneous firing of multiple neurons is common. However, simultaneously firing neurons are usually detected on distinct sets of channels.""",

"""To deal with the problem of temporally overlapping spikes, we therefore sought to detect spikes as local spatiotemporal events ( Fig. 2 ). This step requires knowledge of the probe geometry, which is specified by the user in the form of an adjacency graph ( Fig. 2 a). We illustrate the spike detection process with reference to a small segment of data containing two temporally overlapping but spatially separated spikes.""",

"""The first stage of the algorithm is high-pass filtering the raw data to remove the slow local field potential signal (Butterworth in forward-backward mode; Fig. 2c). Next, spikes are detected using a double-threshold flood fill algorithm ( Fig. 2 d,e). Specifically, spikes are detected as spatiotemporally connected components, in which the filtered signal exceeds a weak threshold θw for every point and in which at least one point exceeds a strong threshold θs. Optimal values for these parameters were found to be 4 and 2 times the s.d. of the filtered signal, as described below.""",

"""Two points are considered neighboring if they are on a single channel and separated by one time sample, or at a single time point on channels joined by the adjacency graph; this allows the algorithm to work with probes of any geometry, not just linear ones. The dual-threshold approach avoids spurious detection of small noise events because isolated islands in which only the weak threshold is exceeded are not retained. Conversely, spikes will not be erroneously split as a result of noise, as areas joined by weak threshold crossings are merged.""",

"""After detection, spikes are temporally realigned to subsample resolution, to the center of mass of the spike’s suprathreshold components, weighted by a power parameter p (see Online Methods). Visual inspection showed that spike times detected with this method corresponded closely to those that would be assigned by a human operator ( Fig. 2 e). The waveforms of each spike are summarized by two vectors.""",

"""First, a feature vector is found by principal component analysis of the realigned waveforms on each channel (three principal components were kept in the analyses reported here). All channels are used in computing the feature vector; thus our two example spikes have similar feature vectors, as their central times are similar ( Fig. 2 f). Second, a mask vector is computed from the peak spike amplitude on each detected channel, rescaled and clipped so channels outside the connected component have mask 0 and channels with amplitude above θs have mask 1. The mask vector allows temporally overlapping spikes to be clustered as coming from separate cells. Indeed, although the feature vectors of our two example spikes were very similar, their mask vectors are completely different ( Fig. 2 g).""",

"""Performance validation and parameter optimization To quantify the performance and optimize the parameters of this algorithm requires ‘ground truth’: knowledge of when the recorded neurons actually fired. We created a simulated ground truth data set by repeatedly adding the spikes of a ‘donor cell’ identified in one recording to a second ‘acceptor’ recording made with same probe. Because the extracellular medium is a linear conductor 26, addition of spike waveforms serves as a sufficient model for overlapping spikes.""",

"""To evaluate the performance of the system, we chose ten donor cells with a variety of amplitudes and waveform distributions ( Fig. 3 a), using recordings from rat cortex with a 32-channel probe shank. To model the variability of waveforms produced by a single neuron due to phenomena such as bursting 27–29, we scaled each spike to a random amplitude in a range that varied by a factor of two (see Online Methods). We refer to the spikes added to the acceptor data set as hybrid spikes and the result as a hybrid data set.""",

"""To evaluate spike detection performance, we used a heuristic criterion to identify which spikes detected by the algorithm corresponded to which hybrid spikes (see Online Methods). We measured performance as a function of three algorithm parameters (θw, θs and p), using four performance statistics. The first statistic was the fraction of hybrid spikes detected ( Fig. 3 b).""",

"""This showed a strong dependence on the thresholds: values of θs above 4 times the s.d. resulted in poor detection, particularly for low-amplitude cells. The dependence of performance on θw was more complex: poor performance resulted not just from overly high values (>2.5 s.d.) but also overly low values (<2 s.d.). Examination of example errors (not shown) indicated that overly low values of θw led to inappropriate merging of temporally overlapping but spatially separated spikes, while overly high values led to artificial splitting of single spikes.""",

"""The second statistic was the total number of detection events ( Fig. 3 c). Because this includes noise events as well as true spikes of the hybrid and background cells, this number should be as small as possible provided the fraction correctly detected remains high. We found that this statistic most critically depended on the strong threshold, increasing markedly for values below 4 s.d.""",

"""The third statistic was timing jitter: the s.d. of the difference between the detected and actual times of each hybrid spike ( Fig. 3 d). Jitter was in all cases less than one sample and improved for larger values of θs and θw, indicating that spike times are best estimated from a minority of larger amplitude spikes. For all hybrid cells, jitter was worse for p < 1; for low amplitude cells, it showed a further worsening for p > 2, reflecting noise introduced by overweighting of peak amplitude times.""",

"""The final statistic was mask accuracy ( Fig. 3 e), which measures how closely the detected mask vectors match those expected from the ground truth (see Online Methods). This showed strongest dependence on θw, with a peak around 2 s.d., and less pronounced dependence on θs, peaking around 5 s.d. We conclude that close to optimal performance can be obtained using a strong threshold of 4 s.d., a weak threshold of 2 s.d. and a power weight of 2. Furthermore, using these parameters yielded around 95% correctly detected spikes and a spike timing jitter of 0.5 samples.""",

"""Cluster analysis The second step of our spike sorting pipeline is automatic cluster analysis, implemented in the program KlustaKwik. For tetrode data, we previously found that fitting a mixture of Gaussians gave close-to-optimal performance 8. This approach cannot be directly ported to high-channel-count data for two reasons.""",

"""The first is the ‘curse of dimensionality’: in high dimensions, noise measured on the large number of uninformative channels will swamp signals measured on the smaller number of informative channels. Second, because temporally overlapping spikes have similar feature vectors ( Fig. 2 f), further information such as the mask vectors must be used to distinguish these spikes. To solve this problem, we designed a new method, the masked EM algorithm 30.""",

"""This algorithm fits the data as a mixture of Gaussians, but with each feature vector replaced by a virtual ensemble in which features with masks near zero are replaced by a noise distribution (see Online Methods). Channels with low mask values are thus ‘disenfranchised’ and do not contribute to cluster assignment; the probabilistic nature of this disenfranchisement means false clusters are not created when amplitudes cross an arbitrary threshold. The computational complexity of this algorithm is better than that of the traditional EM algorithm, scaling with the mean number of unmasked channels per spike (which does not increase for larger arrays) rather than the total number of channels.""",

"""To evaluate the performance of this algorithm, we used the hybrid data sets described above. For each data set, we identified the cluster containing the most hybrid spikes and computed the false discovery rate (fraction of spikes in the cluster that were not hybrids) and the true positive rate (fraction of all hybrid spikes assigned to the cluster). To estimate the theoretical optimum performance that could be expected, we used the best ellipsoid error rate (BEER) measure 8, which fits a quadratic decision boundary using ground truth data and evaluates its performance with cross-validation, varying the parameters of the classifier to obtain a receiver-operating characteristics (ROC) curve showing optimal performance.""",

"""The masked EM algorithm’s performance on an example hybrid data set was close to the optimum estimated by the BEER measure, but the classical EM algorithm’s performance was poor, with error rates typically exceeding 50% ( Fig. 4 a). Across all hybrid data sets, we found no significant difference between the total error of the masked EM algorithm and theoretical optimal performance (P = 0.8, t-test), but a significant difference between the performance of the classical and masked EM algorithms (P = 0.005, t-test; Fig. 4 b).""",

"""To ensure the poor performance of the classical EM algorithm did not simply reflect incorrect parameter choice, we reran it for multiple values of the penalty parameter (which determines the number of clusters found), but this could not improve classical EM performance. This analysis also demonstrated that the error rates of the masked EM algorithm were largely independent of the penalty parameter; using a value corresponding to the Bayesian information criterion seems a good option for penalty choice, as it led to a reasonably small number of clusters without compromising error rates ( Fig. 4 c,d). We conclude that the performance of the masked EM algorithm is close to optimal for this clustering problem, yielding false positive and false discovery rates both on the order of 5%.""",

"""Manual curation The final step of the spike sorting pipeline is manual verification and adjustment of cluster assignments, which are implemented in the program KlustaViewa. Although semiautomatic clustering provides more consistency and lower error rates than fully manual spike sorting 8, further manual corrections are typically required, such as merging of clusters split as a result of electrode drift, bursting or other reasons 27–29.""",

"""These waveform shifts are hard to model and correct mathematically, but can usually be identified by inspection of waveforms, auto- and cross-correlograms, and cluster shapes. It is essential that this step be done with a minimum of human operator time, a particularly acute problem with the very large numbers of neurons recorded by large dense electrode arrays. Specifically, if N clusters are produced automatically, it is impractical for a human operator to inspect all order N2 potential merges.""",

"""We addressed this problem using a semiautomatic ‘wizard’ that reduces the number of potential merges to order N. The wizard works by presenting the operator with pairs of potentially mergeable clusters, ordered by a measure of pairwise cluster similarity. Because the wizard is used iteratively, this measure must be computable in a fraction of a second, even for data sets containing millions of spikes. Thus, only metrics based on summary statistics of each cluster, rather than individual points, are suitable.""",

"""We evaluated several candidate similarity measures. The Kullback-Leibler divergence between two Gaussian distributions was unsuitable as it overweighted differences in covariance matrix relative to differences in the mean. However, we obtained good performance using a single step of the masked EM algorithm to compute the similarity of the mean of one cluster to each of the others ( Fig. 5 a). To verify the accuracy of this measure, we simulated automatic clustering errors by splitting the ground truth clusters in the hybrid data sets into two subclusters, containing high- and low-amplitude spikes.""",

"""In all cases, the similarity measure correctly identified the other half of the artificially split cluster ( Fig. 5 b). The manual stage can take several hours of operator time, and human error is lowest during the start of this period. The wizard therefore iteratively presents the operator with decisions that can be made quickly, with the most important decisions presented first. The wizard iterates through all clusters starting with the best currently unsorted spikes.""",

"""The remaining clusters are ordered by similarity to the best unsorted cluster, and the decision of whether to merge, split or delete each merge candidate is in turn made by the operator ( Fig. 5 c,d). Once satisfied that no more potential merges exist for the currently best unsorted cluster, the operator either accepts it as a well-isolated neuron or rejects it as multiunit activity or noise, and the top-level iteration begins again.""",

"""Although the wizard guides the operator through the decision process, the operator at all times has free access to all data required to make rapid decisions, provided by KlustaViewa’s graphical user interface, designed to be user-friendly and easily navigable ( Fig. 6 ). Using this software, the time taken for manual curation scales linearly with the number of clusters, with a scaling factor that varies between operators and is generally about 1 min per cluster, regardless of probe size. This software therefore allows thorough manual curation of a dense-array recording in a few hours.""",

"""We assessed the performance of eight human operators (five experienced spike sorters, three novices) using this system ( Fig. 7 a). First, we asked whether the operators would correctly fix a misclustering that was produced by the masked EM algorithm in simulation of electrode drift (described further below). All experienced operators and all but one of the novices did this correctly.""",

"""Second we asked how consistent the results of these operators would be on the same data set ( Fig. 7 b–d). We separately assessed consistency on spikes that all operators had identified be in good clusters, on spikes that at least one operator had identified to be in a good cluster, and on all remaining spikes. Similarity was assessed with the Fowlkes-Mallows index 31, which gives a score between 1 for complete agreement and 0 for complete disagreement.""",

"""For all operators apart from one of the novices, consistency was extremely high for those spikes identified as valid by at least one operator ( Fig. 7 e,f); nevertheless, the judgment of whether a cluster should be considered well-isolated varied between operators ( Fig. 7 g). We conclude that experienced operators are likely to make accurate and consistent judgments on cluster merging identification, but that the judgment on which clusters to term valid is inconsistent. We therefore recommend that quantitative metrics 32,33 be used to determine isolation quality.""",

"""Additional tests We used the system described above to answer several more questions regarding the process of spike sorting and the design of electrodes. First, we used our simulated ground truth data set to ask how spike sorting performance would change for different electrode designs.""",

"""We considered two cases. In the first (‘site thinning’; Supplementary Figs. 1 and 2), the electrode was made less dense by omitting alternating channels on both sides. We evaluated the performance of spike detection and clustering using the same hybrid spikes described earlier, but only on this subset of channels. The adjacency graph was modified to join any two channels that both connected to a missing channel. Spike detection was strongly affected, with correct detection rates dropping to an average of below 80% (Supplementary Fig. 1).""",

"""Clustering performance was also impaired, as assessed both by the theoretical optimum and by the masked EM algorithm. While some cells (typically those found on multiple channels) saw little decrease in clustering performance, others were strongly affected by both metrics (Supplementary Fig. 2). We conclude that performance in rat cortex decreases substantially for site spacing larger than the 40-µm same-side site spacing of these test probes.""",

"""Next we simulated removing one side of the probe (Supplementary Figs. 3 and 4). Of the ten hybrid cells analyzed, six were detectable on only one of the probe’s two sides, while the other four could be detected on both sides to a greater or lesser extent (Supplementary Table 1). The effect of side removal was different from that of site thinning. The performance of each unit’s preferred side was comparable to that of the full probe.""",

"""However, for the four units that were visible on both sides of the probe, performance on the unpreferred side was substantially worse than performance on the full probe, as assessed both by theoretical optimum performance and the actual results of the masked EM algorithm. We conclude that, in staggered probes, the probe’s two sides function largely independently: the primary benefit of two-sided shanks is not to increase the isolation quality of a cell already well isolated on one side of the probe, but to record from more units.""",

"""Next we asked whether similar performance to that seen in neocortex could also be obtained in other brain structures and species. We first generated five more hybrid cells using ten-site recordings from the CA1 area of rat hippocampus (Supplementary Figs. 5 and 6). Good performance was again obtained; furthermore, the spike detection parameters found to be optimal in cortical data were also optimal in CA1 data.""",

"""We then ran the same code on high-count data collected from a wider range of preparations: V1 of awake mouse and awake macaque monkey (Supplementary Figs. 7–9) and LGN thalamus of anesthetized marmoset (Supplementary Fig. 10). Additional confidence in the method was provided both by further analyses of hybrid data (Supplementary Fig. 11) and by the observation of sharp orientation-tuned responses (Supplementary Fig. 7 c–l), including among cells of apparently similar waveforms that were nevertheless separated by the spike sorting procedure (Supplementary Fig. 7 m).""",

"""We then asked how well the system would handle non-stationarity in spike amplitudes. Such non-stationarity can occur both because of electrode drift and also because of activity-related changes in spike amplitude such as that after bursts or prolonged periods of firing 27. Examination of data from acute recordings (where electrode drift is often stronger than with chronic probes) showed that the algorithm often tracked drift successfully, but in other cases split the spikes of a single ‘drifty’ cell into multiple clusters requiring manual merging (Supplementary Fig. 12).""",

"""To simulate nonstationarity, we constructed six hybrid data sets in which spike amplitude drifted throughout the recording as a geometric random walk (Supplementary Fig. 13). Spike detection was hardly affected by this nonstationarity (Supplementary Fig. 14). For clustering, only one of the six drifty hybrid data sets required manual curation, and once this was performed, accuracy of the masked EM algorithm was comparable to the theoretical optimum (Supplementary Fig. 15).""",

"""A different type of nonstationarity, in which the hybrid cell simply stopped firing halfway through the recording, also had no effects on performance (P = 0.75; two-sample t-test on total errors; Supplementary Fig. 16). As an important task is often to track cells between recordings made over multiple days—that is, where drift occurs in nonrecorded periods—we also asked whether the wizard’s similarity metric might be used for this purpose. Although ground truth data were not available, a conservative criterion gave encouraging results, as indicated by the similarities of the autocorrelograms of the units associated to each other (Supplementary Fig. 17).""",

"""A strategy sometimes used to deal with nonstationarity is to include time as an additional feature in the cluster analysis algorithm, in principle allowing the algorithm to track slow changes in amplitude. To our surprise, we found that this actually worsened clustering performance, and this worsening could not always be overcome by manual curation (Supplementary Fig. 15). We conclude that nonstationarity (at least of the type modeled here) does not present a serious problem to automatic sorting performance if time is not added as an additional feature and if manual curation is performed when required.""",

"""DISCUSSION We have produced a software suite for spike sorting of data from large, dense electrode arrays. Analysis of simulated ground-truth data indicated that error rates of this approach were frequently of the order 5%.""",

"""A critical step in this system, and all others currently in wide use for in vivo data, is manual curation. Extracellular array recordings are subject to many sources of error, including electrode drift, overlapping spikes and the fact that neuronal spike waveforms are not constant but change according to firing patterns including but not limited to bursting 27–29. While most working neurophysiologists have a good understanding of these potential artifacts, formalizing this knowledge into a reliable mathematical model has proven challenging.""",

"""Because spike sorting errors could lead to erroneous scientific conclusions 29, it remains essential that a scientist is able to inspect the results produced by an automatic algorithm, then correct or discard its results. We found that experienced operators tended to make similar judgments during the manual curation process, but that their judgments of which units were well-isolated were subjective. Fortunately, quantitative criteria exist for assessing the quality of unit isolation 32,33, and we therefore recommend that these be used, rather than human judgments, when deciding which cells to include in further scientific analysis.""",

"""The performance of the system is sufficient for practical analysis of data produced by current commercially available silicon probes. Nevertheless, there remain areas for further improvement. The first of these concerns execution time. KlustaKwik is several orders of magnitude faster than standard mixture-of-Gaussians fitting; nevertheless, when running on large data sets, it can take hours or even days to complete on a standard single-processor machine.""",

"""Hardware acceleration such as GPUs 34 or cloud computing 35 may speed up this analysis stage, as may alternative cluster analysis algorithms that exclude the most computationally expensive step of covariance matrix estimation (for example, refs. 36,37). Faster versions of the code presented here, now under development, will be available at https://github.com/kwikteam/klustakwik2/
 and https://github.com/kwikteam/phy/
. A second opportunity for improvement regards the detection of spatiotemporally overlapping spikes.""",

"""While the current algorithm can detect the majority of temporally overlapping spikes, which occur on distinct sets of channels, it cannot resolve spikes that overlap in both space and time. Template-matching algorithms have solved this problem in the case of in vitro retinal array data 38,39, but these data are much less noisy than in vivo brain recordings. While recent research suggests that certain forms of template matching may succeed, at least for tetrode data in vivo 18,21, such methods are not at present widely applied to in vivo recordings, and many challenges remain to be overcome, most critically regarding the manual curation step.""",

"""The platform we have described here constitutes both a practical solution to today’s spike sorting challenges and also a framework from which to develop solutions for future generations of electrodes containing thousands of channels."""

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
        # Format: [(char_start, char_end, word, tk_start, tk_end), ...]
        self.word_positions = []
        self.current_word_index = -1
        
        # Word instance tracking (for ML training - links EEG to specific word positions)
        # This enables confusion event linking without relying on timestamps
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        
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
        
        Now includes word embeddings for each word position.
        Format: (char_start, char_end, word, tk_start, tk_end, embedding)
        """
        self.word_positions = []
        text = TRAINING_TEXTS[self.parent.current_text_index]
        text_idx = self.parent.current_text_index
        
        # Get word embeddings instance
        embeddings = get_word_embeddings()
        
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
            
            # Get word embedding (uses cache from precompute or computes on demand)
            embedding = embeddings.get_embedding(word)
            
            self.word_positions.append((start, i, word, tk_start, tk_end, embedding))
        
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
                # text_index + char_start/end uniquely identifies the confused word(s)
                event_data = {
                    'text': selected_text,           # The full selected text
                    'words': words,                   # List of individual words
                    'text_index': self.parent.current_text_index,  # Which passage
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
                # text_index + char_start/end uniquely identifies the confused sentence
                event_data = {
                    'text': sentence,
                    'text_index': self.parent.current_text_index,  # Which passage
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
        
        OPTIMIZED: Uses pre-calculated word positions + INTERPOLATION.
        When cursor jumps over words (macOS drops motion events), we fill
        in the skipped words to ensure continuous tracking for ML data.
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
            new_word_idx = self._find_word_at_char(col)
            
            # Only update if we're on a different word
            if new_word_idx >= 0 and new_word_idx != self.current_word_index:
                old_idx = self.current_word_index
                
                # INTERPOLATION: If we jumped FORWARD over words, fill them in
                # This handles dropped motion events during fast reading
                if old_idx >= 0 and new_word_idx > old_idx + 1:
                    for i in range(old_idx + 1, new_word_idx):
                        _, _, skipped_word, _, _, skipped_emb = self.word_positions[i]
                        # Update current_word for each skipped word
                        # This ensures EEG samples get tagged with all words read
                        self.current_word = skipped_word
                        self.parent.current_word = skipped_word
                        self.parent.current_word_embedding = skipped_emb
                        # NOTE: Removed print() here - was blocking I/O on every word
                
                # Now update to the actual current word
                char_start, char_end, word, tk_start, tk_end, embedding = self.word_positions[new_word_idx]
                
                self.current_word = word
                self.current_word_char_start = char_start
                self.current_word_char_end = char_end
                self.parent.current_word = word
                self.parent.current_word_char_start = char_start
                self.parent.current_word_char_end = char_end
                self.parent.current_word_embedding = embedding
                # NOTE: Removed print() here - was causing 5Hz lag due to console I/O
                
                # Update underline: remove old, add new
                if self.current_word_start and self.current_word_end:
                    self.text_display.tag_remove("current_word",
                                                  self.current_word_start,
                                                  self.current_word_end)
                
                self.text_display.tag_add("current_word", tk_start, tk_end)
                self.current_word_start = tk_start
                self.current_word_end = tk_end
                self.current_word_index = new_word_idx
                
                # Update status label
                self.word_status.config(text=f"Current word: {word}")
        except:
            pass
    
    def on_mouse_leave(self, event):
        """Handle mouse leaving text area"""
        self.current_word = ""
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        self.parent.current_word = ""
        self.parent.current_word_char_start = -1
        self.parent.current_word_char_end = -1
        self.parent.current_word_embedding = None  # Clear embedding when leaving
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
        self.current_word_char_start = -1
        self.current_word_char_end = -1
        self.parent.current_word = ""
        self.parent.current_word_char_start = -1
        self.parent.current_word_char_end = -1
        self.parent.current_word_embedding = None  # Clear embedding when changing text
        self.word_status.config(text="Current word: -")
        self.current_word_start = None
        self.current_word_end = None
        self.current_word_index = -1
        # Pre-calculate word positions for fast cursor tracking (including embeddings)
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
        """Low-frequency cursor polling as BACKUP for missed motion events.
        
        Motion events handle most tracking; this is fallback only.
        Reduced from 60Hz to 10Hz to avoid blocking Tkinter event loop.
        """
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
            
            # Reduced polling (10 Hz) - motion events handle most updates
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
        
        # Performance optimization: cached info panel text objects
        self.info_text_objects = {}
        self.info_panel_initialized = False
        
        # Spectral update throttling (every N frames instead of every frame)
        self.spectral_update_counter = 0
        self.spectral_update_interval = 4  # Update spectral every 4 frames (~6Hz)
        
        # Focus-based update skipping: don't waste CPU on hidden/unfocused plot
        self.plot_window_focused = False  # Start False until window opens
        
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
        self.current_word_char_start = -1  # Character position in text (unique word instance ID)
        self.current_word_char_end = -1
        self.current_word_embedding = None  # Word embedding vector for current word
        self.data_collection_paused = False
        
        # Word embeddings (initialized at start)
        self.word_embeddings = None
        self.embedding_dim = 384  # Default, updated when embeddings load
        
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
        # Word instance tracking: each entry is (text_idx, char_start, char_end, word)
        # This enables linking confusion events to specific word instances without timestamps
        self.recorded_word_instances = []
        # Word embeddings for each recorded sample
        self.recorded_word_embeddings = []
        
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
                        # Record word instance: (text_idx, char_start, char_end, word)
                        # This creates a unique ID for each word position, enabling
                        # confusion event linking without relying on timestamps
                        self.recorded_word_instances.append((
                            self.current_text_index,
                            self.current_word_char_start,
                            self.current_word_char_end,
                            self.current_word
                        ))
                        # Record word embedding for this sample
                        # If no word is being tracked, use zero vector
                        if self.current_word_embedding is not None:
                            self.recorded_word_embeddings.append(self.current_word_embedding)
                        else:
                            self.recorded_word_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                    
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
                self.recorded_word_instances = []
                self.recorded_word_embeddings = []
            
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
                    'word_instances': self.recorded_word_instances.copy(),
                    'word_embeddings': self.recorded_word_embeddings.copy(),
                    'embedding_dim': self.embedding_dim,
                    'start_time': self.recording_start_time,
                    'current_text_index': self.current_text_index
                }
                data_points = len(self.recorded_timestamps)
                events_count = len(self.recorded_events)
                unique_words = len(set(wi[3] for wi in self.recorded_word_instances if wi[3]))
            
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
            save_dir = os.path.expanduser("~/Documents/EEGAssistant/NewRecordings")
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
            
            # Word instances: each entry is (text_idx, char_start, char_end, word)
            # This enables linking confusion events to specific word positions
            word_instances = save_data.get('word_instances', [])
            if word_instances:
                word_text_indices = np.array([wi[0] for wi in word_instances], dtype=np.int32)
                word_char_starts = np.array([wi[1] for wi in word_instances], dtype=np.int32)
                word_char_ends = np.array([wi[2] for wi in word_instances], dtype=np.int32)
                words_array = np.array([wi[3] for wi in word_instances], dtype=object)
            else:
                word_text_indices = np.array([], dtype=np.int32)
                word_char_starts = np.array([], dtype=np.int32)
                word_char_ends = np.array([], dtype=np.int32)
                words_array = np.array([], dtype=object)
            
            # Word embeddings: semantic vectors for each sample
            word_embeddings = save_data.get('word_embeddings', [])
            embedding_dim = save_data.get('embedding_dim', 384)
            if word_embeddings:
                word_embeddings_array = np.array(word_embeddings, dtype=np.float32)
            else:
                word_embeddings_array = np.array([], dtype=np.float32).reshape(0, embedding_dim)
            
            # Events - now with structured data including text_index
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
                # Extract text index - which passage the confusion was in
                event_text_indices = np.array([e[2].get('text_index', -1) if isinstance(e[2], dict) else -1 
                                              for e in save_data['events']], dtype=np.int32)
                # Extract character positions for unique word instance identification
                event_char_starts = np.array([e[2].get('char_start', -1) if isinstance(e[2], dict) else -1 
                                             for e in save_data['events']], dtype=np.int32)
                event_char_ends = np.array([e[2].get('char_end', -1) if isinstance(e[2], dict) else -1 
                                           for e in save_data['events']], dtype=np.int32)
            else:
                event_timestamps = np.array([])
                event_types = np.array([])
                event_data_list = np.array([], dtype=object)
                event_texts = np.array([], dtype=object)
                event_words_list = np.array([], dtype=object)
                event_text_indices = np.array([], dtype=np.int32)
                event_char_starts = np.array([], dtype=np.int32)
                event_char_ends = np.array([], dtype=np.int32)
            
            relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
            
            # Metadata
            metadata = {
                'device': 'Muse S Athena',
                'format_version': 3,  # Version 3: includes word embeddings
                'start_time': float(save_data['start_time']),
                'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps),
                'total_events': len(save_data['events']),
                'unique_word_positions': len(set((wi[0], wi[1]) for wi in word_instances if wi[1] >= 0)),
                'eeg_channels': ['TP9', 'AF7', 'AF8', 'TP10'],
                'fnirs_channels': ['Ch1_norm', 'Ch2_norm', 'Ch3_norm', 'Ch4_norm', 
                                 'Ch1_raw', 'Ch2_raw', 'Ch3_raw', 'Ch4_raw'],
                'motion_channels': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                'ref_channels': ['DRL', 'REF'],
                'event_types': ['word_confusion', 'sentence_confusion', 'marker_1', 'marker_2', 'marker_3'],
                'training_texts_count': len(TRAINING_TEXTS),
                'embedding_dim': embedding_dim,
                'embedding_method': self.word_embeddings.method if self.word_embeddings else 'none',
            }
            
            print("Saving to file...")
            np.savez_compressed(
                filename,
                # Timestamps
                timestamps=timestamps,
                relative_timestamps=relative_timestamps,
                # Brain data
                eeg=eeg_data,
                fnirs=fnirs_data,
                motion=motion_data,
                ref=ref_data,
                # Word instance tracking (enables confusion linking)
                word_text_indices=word_text_indices,
                word_char_starts=word_char_starts,
                word_char_ends=word_char_ends,
                words=words_array,
                # Word embeddings (semantic vectors for each sample)
                word_embeddings=word_embeddings_array,
                # Confusion events
                event_timestamps=event_timestamps,
                event_types=event_types,
                event_texts=event_texts,
                event_words_list=event_words_list,
                event_text_indices=event_text_indices,
                event_char_starts=event_char_starts,
                event_char_ends=event_char_ends,
                event_data=event_data_list,
                # Metadata
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
        
        # Prepare data (includes word embeddings)
        save_data = {
            'timestamps': self.recorded_timestamps,
            'eeg': self.recorded_eeg,
            'fnirs': self.recorded_fnirs,
            'motion': self.recorded_motion,
            'ref': self.recorded_ref,
            'events': self.recorded_events,
            'word_instances': self.recorded_word_instances,
            'word_embeddings': self.recorded_word_embeddings,
            'embedding_dim': self.embedding_dim,
            'start_time': self.recording_start_time,
            'current_text_index': self.current_text_index
        }
        
        # Auto-save to same directory as regular saves
        save_dir = os.path.expanduser("~/Documents/EEGAssistant/NewRecordings")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create descriptive filename
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        event_count = len(save_data['events'])
        duration = int(save_data['timestamps'][-1] - save_data['timestamps'][0]) if save_data['timestamps'] else 0
        filename = os.path.join(save_dir, f"eeg_confusion_AUTOSAVE_{timestamp_str}_{duration}s_{event_count}events.npz")
        
        print(f"Auto-saving to: {filename}")
        
        # Quick save
        try:
            # Convert to arrays
            timestamps = np.array(save_data['timestamps'])
            eeg_data = np.array(save_data['eeg']) if save_data['eeg'] else np.array([])
            fnirs_data = np.array(save_data['fnirs']) if save_data['fnirs'] else np.array([])
            motion_data = np.array(save_data['motion']) if save_data['motion'] else np.array([])
            ref_data = np.array(save_data['ref']) if save_data['ref'] else np.array([])
            
            # Word instances: each entry is (text_idx, char_start, char_end, word)
            word_instances = save_data.get('word_instances', [])
            if word_instances:
                word_text_indices = np.array([wi[0] for wi in word_instances], dtype=np.int32)
                word_char_starts = np.array([wi[1] for wi in word_instances], dtype=np.int32)
                word_char_ends = np.array([wi[2] for wi in word_instances], dtype=np.int32)
                words_array = np.array([wi[3] for wi in word_instances], dtype=object)
            else:
                word_text_indices = np.array([], dtype=np.int32)
                word_char_starts = np.array([], dtype=np.int32)
                word_char_ends = np.array([], dtype=np.int32)
                words_array = np.array([], dtype=object)
            
            # Word embeddings
            word_embeddings = save_data.get('word_embeddings', [])
            embedding_dim = save_data.get('embedding_dim', 384)
            if word_embeddings:
                word_embeddings_array = np.array(word_embeddings, dtype=np.float32)
            else:
                word_embeddings_array = np.array([], dtype=np.float32).reshape(0, embedding_dim)
            
            # Events - with structured data including text_index
            if save_data['events']:
                event_timestamps = np.array([e[0] for e in save_data['events']])
                event_types = np.array([e[1] for e in save_data['events']])
                event_data_list = np.array([e[2] for e in save_data['events']], dtype=object)
                event_texts = np.array([e[2].get('text', '') if isinstance(e[2], dict) else e[2] 
                                       for e in save_data['events']], dtype=object)
                event_words_list = np.array([e[2].get('words', []) if isinstance(e[2], dict) else [] 
                                            for e in save_data['events']], dtype=object)
                event_text_indices = np.array([e[2].get('text_index', -1) if isinstance(e[2], dict) else -1 
                                              for e in save_data['events']], dtype=np.int32)
                event_char_starts = np.array([e[2].get('char_start', -1) if isinstance(e[2], dict) else -1 
                                             for e in save_data['events']], dtype=np.int32)
                event_char_ends = np.array([e[2].get('char_end', -1) if isinstance(e[2], dict) else -1 
                                           for e in save_data['events']], dtype=np.int32)
            else:
                event_timestamps = np.array([])
                event_types = np.array([])
                event_data_list = np.array([], dtype=object)
                event_texts = np.array([], dtype=object)
                event_words_list = np.array([], dtype=object)
                event_text_indices = np.array([], dtype=np.int32)
                event_char_starts = np.array([], dtype=np.int32)
                event_char_ends = np.array([], dtype=np.int32)
            
            relative_timestamps = timestamps - timestamps[0] if len(timestamps) > 0 else np.array([])
            
            metadata = {
                'device': 'Muse S Athena',
                'format_version': 3,  # Version 3: includes word embeddings
                'start_time': float(save_data['start_time']),
                'end_time': float(timestamps[-1] if len(timestamps) > 0 else save_data['start_time']),
                'duration': float(timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0),
                'sample_rate': self.sample_rate,
                'total_samples': len(timestamps),
                'total_events': len(save_data['events']),
                'unique_word_positions': len(set((wi[0], wi[1]) for wi in word_instances if wi[1] >= 0)),
                'training_texts_count': len(TRAINING_TEXTS),
                'embedding_dim': embedding_dim,
                'embedding_method': self.word_embeddings.method if self.word_embeddings else 'none',
            }
            
            np.savez_compressed(
                filename,
                timestamps=timestamps,
                relative_timestamps=relative_timestamps,
                eeg=eeg_data,
                fnirs=fnirs_data,
                motion=motion_data,
                ref=ref_data,
                word_text_indices=word_text_indices,
                word_char_starts=word_char_starts,
                word_char_ends=word_char_ends,
                words=words_array,
                word_embeddings=word_embeddings_array,
                event_timestamps=event_timestamps,
                event_types=event_types,
                event_texts=event_texts,
                event_words_list=event_words_list,
                event_text_indices=event_text_indices,
                event_char_starts=event_char_starts,
                event_char_ends=event_char_ends,
                event_data=event_data_list,
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
        # Title now set in _init_info_panel() to avoid recreating every frame
        
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
        """Update all plots (skipped when window not focused for performance)"""
        # Skip expensive visualization when plot window isn't visible/focused
        # Data collection continues in background via receiver_loop
        if not self.plot_window_focused:
            return list(self.lines.values()) + list(self.spectral_lines.values())
        
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
            
            # Update spectral analysis (throttled - expensive FFT)
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
    
    def _init_info_panel(self):
        """Initialize info panel text objects ONCE (not every frame)"""
        ax = self.axes['info']
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        t = self.info_text_objects  # shorthand
        y = 0.95
        
        # Static title
        ax.text(0.1, y, 'Signal Statistics', fontsize=12, weight='bold', 
               color='#FFD93D', transform=ax.transAxes)
        
        # Recording status (dynamic)
        y -= 0.06
        t['rec_status'] = ax.text(0.1, y, '', fontsize=10, color='#ff4444', 
                                  weight='bold', transform=ax.transAxes)
        y -= 0.04
        t['samples'] = ax.text(0.1, y, '', fontsize=9, color='#ff6666', transform=ax.transAxes)
        y -= 0.04
        t['word_conf'] = ax.text(0.1, y, '', fontsize=9, color='#ffaa44', transform=ax.transAxes)
        y -= 0.04
        t['sent_conf'] = ax.text(0.1, y, '', fontsize=9, color='#ff8844', transform=ax.transAxes)
        
        # Current word
        y -= 0.06
        t['word'] = ax.text(0.1, y, 'Word: -', fontsize=9, color='#FFD93D', transform=ax.transAxes)
        y -= 0.04
        t['text_idx'] = ax.text(0.1, y, f'Text: 1/{len(TRAINING_TEXTS)}', fontsize=9, 
                               color='#aaaaaa', transform=ax.transAxes)
        
        # EEG header
        y -= 0.08
        ax.text(0.1, y, 'EEG (4ch):', fontsize=10, weight='bold', color='#4ECDC4', transform=ax.transAxes)
        y -= 0.05
        
        # EEG channels
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            ax.text(0.15, y, f'{ch}:', fontsize=9, color=self.eeg_colors[ch], transform=ax.transAxes)
            t[f'eeg_{ch}'] = ax.text(0.4, y, '0.0±0.0', fontsize=9, color='white', transform=ax.transAxes)
            y -= 0.04
        
        # System header
        y -= 0.06
        ax.text(0.1, y, 'System:', fontsize=10, weight='bold', color='#95A5A6', transform=ax.transAxes)
        y -= 0.05
        
        # System stats
        for label, color in [('Packets:', 'white'), ('EEG:', '#4ECDC4'), ('fNIRS:', '#E74C3C')]:
            ax.text(0.15, y, label, fontsize=9, color='white', transform=ax.transAxes)
            t[label] = ax.text(0.4, y, '0', fontsize=9, color=color, transform=ax.transAxes)
            y -= 0.04
        
        self.info_panel_initialized = True
    
    def _update_info_panel(self):
        """Update statistics panel - only updates text content, not structure"""
        # Initialize panel structure once
        if not self.info_panel_initialized:
            self._init_info_panel()
        
        t = self.info_text_objects
        
        # Recording status
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            status_text = f'⏺ REC: {elapsed:.1f}s'
            if self.data_collection_paused:
                status_text += ' (PAUSED)'
            t['rec_status'].set_text(status_text)
            t['samples'].set_text(f'Samples: {len(self.recorded_timestamps)}')
            
            word_confusion = sum(1 for _, et, _ in self.recorded_events if et == 'word_confusion')
            sentence_confusion = sum(1 for _, et, _ in self.recorded_events if et == 'sentence_confusion')
            t['word_conf'].set_text(f'🤔 Word: {word_confusion}')
            t['sent_conf'].set_text(f'📄 Sentence: {sentence_confusion}')
        else:
            t['rec_status'].set_text('')
            t['samples'].set_text('')
            t['word_conf'].set_text('')
            t['sent_conf'].set_text('')
        
        # Current word
        t['word'].set_text(f'Word: {self.current_word[:15] if self.current_word else "-"}')
        t['text_idx'].set_text(f'Text: {self.current_text_index + 1}/{len(TRAINING_TEXTS)}')
        
        # EEG stats
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            if len(self.eeg_channels[ch]) > 0:
                data = np.array(list(self.eeg_channels[ch])[-100:])
                t[f'eeg_{ch}'].set_text(f'{np.mean(data):.1f}±{np.std(data):.1f}')
        
        # System stats
        t['Packets:'].set_text(str(self.packet_count))
        t['EEG:'].set_text(str(self.eeg_packet_count))
        t['fNIRS:'].set_text(str(self.fnirs_packet_count))
    
    def start(self):
        """Start the visualizer"""
        print("\n" + "="*60)
        print("   MUSE S ATHENA - CONFUSION DETECTION WITH TEXT SELECTION")
        print("="*60)
        
        # Initialize word embeddings (before teleprompter so words can be embedded)
        print("\n🧠 Initializing word embeddings...")
        self.word_embeddings = get_word_embeddings()
        self.embedding_dim = self.word_embeddings.embedding_dim
        
        # Pre-compute embeddings for all training texts
        print("   Pre-computing embeddings for training texts...")
        self.word_embeddings.precompute_text_embeddings(TRAINING_TEXTS)
        print(f"   Embedding dimension: {self.embedding_dim}")
        
        print(f"\n📡 Listening for OSC data on UDP port {self.port}")
        print("\n🖱️ IMPROVED SELECTION SYSTEM:")
        print("  • LEFT-DRAG: Select multi-word phrases that confuse you")
        print("  • RIGHT-CLICK: Mark entire sentence as confusing")
        print("  • C: Toggle between reading and labeling modes")
        print("\n📊 Data saves with full text selections + word embeddings for ML training")
        
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
        
        # Focus tracking: skip expensive updates when plot window not visible
        def on_plot_focus_in(event):
            self.plot_window_focused = True
        
        def on_plot_focus_out(event):
            self.plot_window_focused = False
        
        # Bind focus events to matplotlib's Tk window
        try:
            plot_window = self.fig.canvas.manager.window
            plot_window.bind('<FocusIn>', on_plot_focus_in)
            plot_window.bind('<FocusOut>', on_plot_focus_out)
            # Also handle window mapping (minimize/restore)
            plot_window.bind('<Map>', on_plot_focus_in)
            plot_window.bind('<Unmap>', on_plot_focus_out)
        except Exception as e:
            print(f"Could not bind focus events: {e}")
            self.plot_window_focused = True  # Fallback: always update
        
        # Close handler
        def on_close(event):
            if self.teleprompter and self.teleprompter.active:
                self.teleprompter.on_close()
            self.cleanup_on_exit()
        
        self.fig.canvas.mpl_connect('close_event', on_close)
        
        # Start animation (reduced from 25 FPS to ~12 FPS for better responsiveness)
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=80,  # ~12 FPS - sufficient for EEG viz, frees CPU for Tkinter
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
            # Skips drawing entirely when plot window isn't focused (huge CPU savings)
            def update_matplotlib():
                try:
                    if self.fig and plt.fignum_exists(self.fig.number):
                        # Only redraw if plot window has focus - saves major CPU
                        if self.plot_window_focused:
                            self.fig.canvas.draw_idle()
                            self.fig.canvas.flush_events()
                except:
                    pass
                if self.teleprompter and self.teleprompter.active:
                    self.teleprompter.root.after(80, update_matplotlib)  # ~12 FPS
            
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