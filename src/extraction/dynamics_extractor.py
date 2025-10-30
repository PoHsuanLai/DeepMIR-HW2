"""
Dynamics Extraction for Music Generation Control
Extracts loudness, RMS energy, and dynamic variations
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class DynamicsExtractor:
    """
    Extract dynamics features for controllable music generation
    """

    def __init__(self, sr: int = 44100):
        """
        Initialize dynamics extractor

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

    def extract_rms(self, audio_path: str, frame_length: int = 2048,
                    hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract RMS energy

        Args:
            audio_path: Path to audio file
            frame_length: Frame length for RMS computation
            hop_length: Hop length between frames

        Returns:
            Tuple of (times, rms_values)
        """
        y, sr = self.load_audio(audio_path)

        # Compute RMS
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Time array
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length
        )

        return times, rms

    def extract_loudness(self, audio_path: str, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract perceptual loudness (A-weighted)

        Args:
            audio_path: Path to audio file
            hop_length: Hop length between frames

        Returns:
            Tuple of (times, loudness_values)
        """
        y, sr = self.load_audio(audio_path)

        # Compute STFT
        D = librosa.stft(y, hop_length=hop_length)

        # A-weighting
        freqs = librosa.fft_frequencies(sr=sr)
        a_weighting = librosa.A_weighting(freqs)

        # Apply A-weighting and compute magnitude
        D_weighted = np.abs(D) * (10 ** (a_weighting[:, np.newaxis] / 20))

        # Compute loudness as mean across frequency
        loudness = np.mean(D_weighted, axis=0)

        # Time array
        times = librosa.frames_to_time(
            np.arange(len(loudness)),
            sr=sr,
            hop_length=hop_length
        )

        return times, loudness

    def extract_dynamic_range(self, audio_path: str, window_size: float = 3.0) -> Dict:
        """
        Compute dynamic range over time

        Args:
            audio_path: Path to audio file
            window_size: Window size in seconds for computing range

        Returns:
            Dictionary with dynamic range information
        """
        y, sr = self.load_audio(audio_path)

        # RMS with short hop
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Compute rolling max and min
        window_frames = int(window_size * sr / 512)

        rolling_max = np.array([
            np.max(rms_db[max(0, i - window_frames):i + 1])
            for i in range(len(rms_db))
        ])

        rolling_min = np.array([
            np.min(rms_db[max(0, i - window_frames):i + 1])
            for i in range(len(rms_db))
        ])

        dynamic_range = rolling_max - rolling_min

        # Overall statistics
        overall_dynamic_range = np.max(rms_db) - np.min(rms_db)
        avg_dynamic_range = np.mean(dynamic_range)

        return {
            'times': times,
            'rms_db': rms_db,
            'dynamic_range': dynamic_range,
            'overall_dynamic_range': float(overall_dynamic_range),
            'avg_dynamic_range': float(avg_dynamic_range)
        }

    def extract_envelope(self, audio_path: str, method: str = 'rms') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitude envelope

        Args:
            audio_path: Path to audio file
            method: Envelope extraction method ('rms' or 'peak')

        Returns:
            Tuple of (times, envelope)
        """
        y, sr = self.load_audio(audio_path)

        if method == 'rms':
            envelope = librosa.feature.rms(y=y, hop_length=512)[0]
        elif method == 'peak':
            # Peak envelope using Hilbert transform
            from scipy.signal import hilbert
            analytic_signal = hilbert(y)
            envelope_full = np.abs(analytic_signal)

            # Downsample to match frame rate
            hop_length = 512
            envelope = librosa.util.sync(
                envelope_full[np.newaxis, :],
                np.arange(0, len(envelope_full), hop_length)
            )[0]
        else:
            envelope = librosa.feature.rms(y=y, hop_length=512)[0]

        times = librosa.frames_to_time(
            np.arange(len(envelope)),
            sr=sr,
            hop_length=512
        )

        return times, envelope

    def create_dynamics_curve(self, audio_path: str, frame_rate: int = 50,
                             smoothing_window: int = 5) -> np.ndarray:
        """
        Create smooth dynamics curve for MuseControlLite

        Args:
            audio_path: Path to audio file
            frame_rate: Target frame rate (Hz)
            smoothing_window: Window size for smoothing

        Returns:
            Dynamics curve array (normalized 0-1)
        """
        y, sr = self.load_audio(audio_path)
        duration = len(y) / sr

        # Extract RMS
        rms = librosa.feature.rms(y=y, hop_length=int(sr / frame_rate))[0]

        # Smooth
        if smoothing_window > 1:
            from scipy.ndimage import uniform_filter1d
            rms = uniform_filter1d(rms, size=smoothing_window, mode='nearest')

        # Normalize to 0-1 range
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)

        return rms_norm

    def extract_dynamics_features(self, audio_path: str) -> Dict:
        """
        Extract comprehensive dynamics information

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with all dynamics features
        """
        # RMS
        rms_times, rms = self.extract_rms(audio_path)

        # Loudness
        loud_times, loudness = self.extract_loudness(audio_path)

        # Dynamic range
        dynamic_range_info = self.extract_dynamic_range(audio_path)

        # Envelope
        env_times, envelope = self.extract_envelope(audio_path, method='rms')

        # Dynamics curve (normalized)
        dynamics_curve = self.create_dynamics_curve(audio_path)

        # Overall statistics
        y, sr = self.load_audio(audio_path)
        overall_rms = np.sqrt(np.mean(y ** 2))
        peak_amplitude = np.max(np.abs(y))
        crest_factor = peak_amplitude / (overall_rms + 1e-8)

        return {
            'rms_times': rms_times,
            'rms': rms,
            'loudness_times': loud_times,
            'loudness': loudness,
            'dynamic_range': dynamic_range_info,
            'envelope_times': env_times,
            'envelope': envelope,
            'dynamics_curve': dynamics_curve,
            'overall_rms': float(overall_rms),
            'peak_amplitude': float(peak_amplitude),
            'crest_factor': float(crest_factor)
        }

    def save_dynamics_features(self, audio_path: str, output_dir: str):
        """
        Extract and save all dynamics features

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(audio_path).stem

        # Extract features
        features = self.extract_dynamics_features(audio_path)

        # Save as NPZ
        output_file = output_dir / f"{filename}_dynamics.npz"
        np.savez(
            output_file,
            rms_times=features['rms_times'],
            rms=features['rms'],
            loudness_times=features['loudness_times'],
            loudness=features['loudness'],
            dynamic_range_times=features['dynamic_range']['times'],
            dynamic_range=features['dynamic_range']['dynamic_range'],
            overall_dynamic_range=features['dynamic_range']['overall_dynamic_range'],
            envelope_times=features['envelope_times'],
            envelope=features['envelope'],
            dynamics_curve=features['dynamics_curve'],
            overall_rms=features['overall_rms'],
            peak_amplitude=features['peak_amplitude'],
            crest_factor=features['crest_factor']
        )

        print(f"Dynamics features saved to {output_file}")

        return features


if __name__ == "__main__":
    # Test dynamics extractor
    extractor = DynamicsExtractor()

    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(test_file).exists():
        print(f"Testing with {test_file}")

        # Extract features
        features = extractor.extract_dynamics_features(test_file)

        print(f"RMS shape: {features['rms'].shape}")
        print(f"Dynamics curve shape: {features['dynamics_curve'].shape}")
        print(f"Overall RMS: {features['overall_rms']:.4f}")
        print(f"Overall dynamic range: {features['dynamic_range']['overall_dynamic_range']:.2f} dB")

        # Save features
        extractor.save_dynamics_features(test_file, "./outputs/dynamics_features")
