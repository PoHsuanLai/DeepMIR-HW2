"""
Rhythm Extraction for Music Generation Control
Extracts beat times, downbeats, tempo, and rhythm patterns
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class RhythmExtractor:
    """
    Extract rhythm features for controllable music generation
    """

    def __init__(self, sr: int = 44100):
        """
        Initialize rhythm extractor

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

    def load_audio(self, audio_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file

        Args:
            audio_path: Path to audio file
            duration: Duration to load (None for full)

        Returns:
            Audio array and sample rate
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
        return y, sr

    def extract_beats(self, audio_path: str) -> Dict:
        """
        Extract beat information using librosa

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with beat information
        """
        y, sr = self.load_audio(audio_path)

        # Extract tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

        # Estimate time signature from beat intervals
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            avg_beat_interval = np.median(beat_intervals)
        else:
            avg_beat_interval = 60.0 / tempo if tempo > 0 else 0.5

        return {
            'tempo': float(tempo),
            'beat_times': beat_times,
            'beat_frames': beat_frames,
            'avg_beat_interval': float(avg_beat_interval),
            'num_beats': len(beat_times)
        }

    def extract_onset_envelope(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract onset strength envelope

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (times, onset_envelope)
        """
        y, sr = self.load_audio(audio_path)

        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        # Time array
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)

        return times, onset_env

    def extract_tempogram(self, audio_path: str) -> Dict:
        """
        Extract tempogram (tempo variation over time)

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with tempogram information
        """
        y, sr = self.load_audio(audio_path)

        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        # Compute tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512
        )

        # Estimate tempo over time
        tempo_times = librosa.frames_to_time(
            np.arange(tempogram.shape[1]),
            sr=sr,
            hop_length=512
        )

        return {
            'tempogram': tempogram,
            'tempo_times': tempo_times,
            'onset_envelope': onset_env
        }

    def create_beat_mask(self, audio_path: str, frame_rate: int = 50) -> np.ndarray:
        """
        Create binary beat mask for MuseControlLite

        Args:
            audio_path: Path to audio file
            frame_rate: Target frame rate (Hz)

        Returns:
            Binary beat mask array
        """
        y, sr = self.load_audio(audio_path)
        duration = len(y) / sr

        # Get beat times
        beat_info = self.extract_beats(audio_path)
        beat_times = beat_info['beat_times']

        # Create mask
        num_frames = int(duration * frame_rate)
        beat_mask = np.zeros(num_frames)

        # Mark beat positions
        for beat_time in beat_times:
            frame_idx = int(beat_time * frame_rate)
            if frame_idx < num_frames:
                beat_mask[frame_idx] = 1.0

        return beat_mask

    def extract_rhythm_features(self, audio_path: str) -> Dict:
        """
        Extract comprehensive rhythm information

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with all rhythm features
        """
        # Beat info
        beat_info = self.extract_beats(audio_path)

        # Onset envelope
        onset_times, onset_env = self.extract_onset_envelope(audio_path)

        # Tempogram
        tempogram_info = self.extract_tempogram(audio_path)

        # Beat mask
        beat_mask = self.create_beat_mask(audio_path)

        return {
            'tempo': beat_info['tempo'],
            'beat_times': beat_info['beat_times'],
            'onset_times': onset_times,
            'onset_envelope': onset_env,
            'tempogram': tempogram_info['tempogram'],
            'beat_mask': beat_mask
        }

    def save_rhythm_features(self, audio_path: str, output_dir: str):
        """
        Extract and save all rhythm features

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(audio_path).stem

        # Extract features
        features = self.extract_rhythm_features(audio_path)

        # Save as NPZ
        output_file = output_dir / f"{filename}_rhythm.npz"
        np.savez(
            output_file,
            tempo=features['tempo'],
            beat_times=features['beat_times'],
            onset_times=features['onset_times'],
            onset_envelope=features['onset_envelope'],
            tempogram=features['tempogram'],
            beat_mask=features['beat_mask']
        )

        print(f"Rhythm features saved to {output_file}")

        return features


if __name__ == "__main__":
    # Test rhythm extractor
    extractor = RhythmExtractor()

    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(test_file).exists():
        print(f"Testing with {test_file}")

        # Extract features
        features = extractor.extract_rhythm_features(test_file)

        print(f"Tempo: {features['tempo']:.2f} BPM")
        print(f"Number of beats: {len(features['beat_times'])}")
        print(f"Beat mask shape: {features['beat_mask'].shape}")

        # Save features
        extractor.save_rhythm_features(test_file, "./outputs/rhythm_features")
