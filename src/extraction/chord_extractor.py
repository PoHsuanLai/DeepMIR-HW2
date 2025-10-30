"""
Chord Extraction for JASCO Conditioning
Uses madmom for advanced chord recognition
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import librosa
import tempfile
import soundfile as sf


class ChordExtractor:
    """
    Extract chord progressions from audio for JASCO conditioning using madmom
    """

    def __init__(self, sr: int = 44100, use_madmom: bool = True):
        """
        Initialize chord extractor

        Args:
            sr: Sample rate for audio processing
            use_madmom: Use madmom for chord recognition (default: True)
        """
        self.sr = sr
        self.use_madmom = use_madmom

        # Initialize madmom chord recognition
        if self.use_madmom:
            try:
                # Use CNNChordFeatureProcessor which is simpler and works better
                from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

                # Create a pipeline: audio -> CNN features -> CRF chord recognition
                self.feature_processor = CNNChordFeatureProcessor()
                self.chord_processor = CRFChordRecognitionProcessor()
                print("madmom chord recognition loaded successfully")
            except ImportError as e:
                print(f"Warning: madmom not available ({e}), falling back to template matching")
                self.use_madmom = False
                self.chord_processor = None
                self.feature_processor = None
        else:
            self.chord_processor = None
            self.feature_processor = None

    def extract_chords_madmom(self, audio_path: str) -> List[Tuple[str, float]]:
        """
        Extract chords using madmom's deep learning model

        Args:
            audio_path: Path to audio file

        Returns:
            List of (chord_name, timestamp) tuples
        """
        if not self.use_madmom or self.chord_processor is None:
            return self.extract_chords_simple(audio_path)

        try:
            # Step 1: Extract CNN chord features from audio
            features = self.feature_processor(audio_path)

            # Step 2: Recognize chords from features using CRF
            chords = self.chord_processor(features)

            # Convert to list of (chord, timestamp) tuples
            # CRFChordRecognitionProcessor returns structured array with fields:
            # 'start', 'end', 'label'
            chord_sequence = []

            # Check if it's a structured array
            if hasattr(chords, 'dtype') and chords.dtype.names:
                # Structured array with named fields
                for chord_event in chords:
                    timestamp = float(chord_event['start'])
                    chord_label = str(chord_event['label'])
                    chord_sequence.append((chord_label, timestamp))
            else:
                # Regular array format (fallback)
                prev_chord = None
                for i in range(len(chords)):
                    timestamp = float(chords[i][0])
                    chord_label = str(chords[i][1])
                    if chord_label != prev_chord:
                        chord_sequence.append((chord_label, timestamp))
                        prev_chord = chord_label

            return chord_sequence

        except Exception as e:
            print(f"Warning: madmom chord extraction failed ({e}), falling back to simple method")
            import traceback
            traceback.print_exc()
            return self.extract_chords_simple(audio_path)

    def extract_chords_simple(self, audio_path: str, hop_length: int = 512) -> List[Tuple[str, float]]:
        """
        Extract simple chord progression using chromagram-based approach

        Args:
            audio_path: Path to audio file
            hop_length: Hop length for analysis

        Returns:
            List of (chord_name, timestamp) tuples
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)

        # Extract chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

        # Smooth chromagram
        chroma_smooth = librosa.decompose.nn_filter(
            chroma,
            aggregate=np.median,
            metric='cosine'
        )

        # Define chord templates (major and minor triads)
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'G#': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'A#': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            # Minor chords
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            'Bm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        }

        # Convert templates to numpy array
        templates = np.array([chord_templates[chord] for chord in chord_templates.keys()])
        chord_names = list(chord_templates.keys())

        # Recognize chords for each frame
        chord_indices = []
        for frame_idx in range(chroma_smooth.shape[1]):
            frame_chroma = chroma_smooth[:, frame_idx]
            # Compute correlation with each template
            correlations = np.dot(templates, frame_chroma)
            # Get best matching chord
            best_chord_idx = np.argmax(correlations)
            chord_indices.append(best_chord_idx)

        # Convert frame indices to chord sequence with timestamps
        # Segment by detecting chord changes
        chord_sequence = []
        current_chord_idx = chord_indices[0]
        start_time = 0.0

        for frame_idx, chord_idx in enumerate(chord_indices):
            if chord_idx != current_chord_idx or frame_idx == len(chord_indices) - 1:
                # Chord changed or end reached
                timestamp = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)
                chord_name = chord_names[current_chord_idx]
                chord_sequence.append((chord_name, start_time))

                current_chord_idx = chord_idx
                start_time = timestamp

        return chord_sequence

    def extract_chords_from_audio(self, audio: np.ndarray, sr: int = 44100) -> List[Tuple[str, float]]:
        """
        Extract chord progression from audio (full mix or any stem)

        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate

        Returns:
            List of (chord_name, timestamp) tuples
        """
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)

        try:
            # Extract chords using madmom if available, otherwise fall back
            if self.use_madmom:
                chords = self.extract_chords_madmom(temp_path)
            else:
                chords = self.extract_chords_simple(temp_path)
        finally:
            # Clean up
            import os
            os.unlink(temp_path)

        return chords

    def extract_chords_from_bass(self, bass_audio: np.ndarray, sr: int = 44100) -> List[Tuple[str, float]]:
        """
        DEPRECATED: Use extract_chords_from_audio instead
        Kept for backward compatibility

        Args:
            bass_audio: Bass audio array
            sr: Sample rate

        Returns:
            List of (chord_name, timestamp) tuples
        """
        return self.extract_chords_from_audio(bass_audio, sr)

    def format_for_jasco(self, chord_sequence: List[Tuple[str, float]], duration: float = 10.0) -> List[Tuple[str, float]]:
        """
        Format chord sequence for JASCO input

        Args:
            chord_sequence: List of (chord, timestamp) tuples
            duration: Target duration in seconds

        Returns:
            Formatted chord sequence for JASCO
        """
        # JASCO expects chords at regular intervals
        # Simplify by taking one chord per second or every 2 seconds

        if not chord_sequence:
            return [('C', 0.0)]  # Default to C major if no chords detected

        # JASCO only supports root notes (no chord qualities like minor, major, etc.)
        # Valid chords: N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        def simplify_chord_for_jasco(chord: str) -> str:
            """Convert chord names to JASCO-compatible root notes"""
            # Handle N (no chord)
            if chord == 'N' or chord == 'X':
                return 'N'

            # Already in JASCO format
            if chord in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                return chord

            # madmom uses different notation, e.g., "C:maj", "C:min", "C:maj7"
            # Split by colon to get root note
            if ':' in chord:
                root = chord.split(':')[0]
                # Handle flats (madmom uses 'b' for flats)
                if 'b' in root:
                    # Convert flats to sharps for JASCO
                    flat_to_sharp = {
                        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
                    }
                    if root in flat_to_sharp:
                        return flat_to_sharp[root]
                if root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                    return root

            # Handle simple minor chords (e.g., "Cm")
            if chord.endswith('m') and len(chord) >= 2:
                root = chord[:-1]
                if root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                    return root

            # Try to extract root note from beginning
            for root in ['C#', 'D#', 'F#', 'G#', 'A#']:  # Check 2-char sharps first
                if chord.startswith(root):
                    return root
            for root in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:  # Then 1-char roots
                if chord.startswith(root):
                    return root

            # Handle flats in root position
            flat_to_sharp = {
                'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
            }
            for flat, sharp in flat_to_sharp.items():
                if chord.startswith(flat):
                    return sharp

            # Default fallback
            return 'C'

        formatted_chords = []
        current_time = 0.0
        interval = 2.0  # Chord change every 2 seconds

        while current_time < duration:
            # Find the chord at this time
            chord_at_time = 'C'  # Default
            for chord, timestamp in chord_sequence:
                if timestamp <= current_time:
                    chord_at_time = simplify_chord_for_jasco(chord)
                else:
                    break

            formatted_chords.append((chord_at_time, current_time))
            current_time += interval

        return formatted_chords

    def extract_and_format_for_jasco(self, audio_path: str, duration: float = 10.0) -> List[Tuple[str, float]]:
        """
        Complete pipeline: extract chords and format for JASCO

        Args:
            audio_path: Path to audio file
            duration: Target duration

        Returns:
            JASCO-formatted chord sequence
        """
        chords = self.extract_chords_simple(audio_path)
        return self.format_for_jasco(chords, duration)
