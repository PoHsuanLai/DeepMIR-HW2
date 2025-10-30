"""
Melody Extraction for Music Generation Control
Supports CQT top-4, Chroma, and F0 extraction
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class MelodyExtractor:
    """
    Extract melody features for controllable music generation
    """

    def __init__(self, sr: int = 44100):
        """
        Initialize melody extractor

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

    def extract_chroma(self, audio_path: str, n_chroma: int = 12) -> np.ndarray:
        """
        Extract chromagram features

        Args:
            audio_path: Path to audio file
            n_chroma: Number of chroma bins

        Returns:
            Chromagram array (n_chroma, n_frames)
        """
        y, sr = self.load_audio(audio_path)

        # Compute chromagram
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_chroma=n_chroma,
            hop_length=512
        )

        return chroma

    def extract_cqt_top4(self, audio_path: str, n_bins: int = 84,
                        bins_per_octave: int = 12) -> np.ndarray:
        """
        Extract CQT top-4 bins (for MuseControlLite)

        Args:
            audio_path: Path to audio file
            n_bins: Number of CQT bins
            bins_per_octave: Bins per octave

        Returns:
            Array of top-4 bin indices per frame (4, n_frames)
        """
        y, sr = self.load_audio(audio_path)

        # Compute CQT
        cqt = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=512,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )

        # Get magnitude
        cqt_mag = np.abs(cqt)

        # Get top-4 indices for each frame
        top4_indices = np.argsort(cqt_mag, axis=0)[-4:]

        return top4_indices

    def extract_f0(self, audio_path: str, fmin: float = 65.0,
                  fmax: float = 2093.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency (F0) contour

        Args:
            audio_path: Path to audio file
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)

        Returns:
            Tuple of (times, f0_values)
        """
        y, sr = self.load_audio(audio_path)

        # Extract F0 using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=512
        )

        # Create time array
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=sr,
            hop_length=512
        )

        return times, f0

    def extract_melody_contour(self, audio_path: str) -> dict:
        """
        Extract comprehensive melody information

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with melody features
        """
        y, sr = self.load_audio(audio_path)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

        # CQT top-4
        cqt = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84)
        cqt_mag = np.abs(cqt)
        top4_indices = np.argsort(cqt_mag, axis=0)[-4:]

        # F0
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=65.0, fmax=2093.0, sr=sr, hop_length=512
            )
        except:
            f0 = np.zeros(chroma.shape[1])
            voiced_flag = np.zeros(chroma.shape[1], dtype=bool)
            voiced_probs = np.zeros(chroma.shape[1])

        # Time array
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr,
            hop_length=512
        )

        return {
            'chroma': chroma,
            'cqt_top4': top4_indices,
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'times': times,
            'sr': sr
        }

    def save_melody_features(self, audio_path: str, output_dir: str):
        """
        Extract and save all melody features

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save features
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(audio_path).stem

        # Extract features
        features = self.extract_melody_contour(audio_path)

        # Save as NPZ
        output_file = output_dir / f"{filename}_melody.npz"
        np.savez(
            output_file,
            chroma=features['chroma'],
            cqt_top4=features['cqt_top4'],
            f0=features['f0'],
            voiced_flag=features['voiced_flag'],
            voiced_probs=features['voiced_probs'],
            times=features['times'],
            sr=features['sr']
        )

        print(f"Melody features saved to {output_file}")

        return features

    def visualize_melody(self, audio_path: str, output_file: Optional[str] = None):
        """
        Visualize melody features

        Args:
            audio_path: Path to audio file
            output_file: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt

            features = self.extract_melody_contour(audio_path)

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            # Chromagram
            librosa.display.specshow(
                features['chroma'],
                y_axis='chroma',
                x_axis='time',
                hop_length=512,
                sr=features['sr'],
                ax=axes[0]
            )
            axes[0].set_title('Chromagram')
            axes[0].label_outer()

            # CQT with top-4 marked
            y, sr = self.load_audio(audio_path)
            cqt = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84)
            librosa.display.specshow(
                librosa.amplitude_to_db(np.abs(cqt), ref=np.max),
                y_axis='cqt_note',
                x_axis='time',
                hop_length=512,
                sr=sr,
                ax=axes[1]
            )
            axes[1].set_title('CQT (Constant-Q Transform)')
            axes[1].label_outer()

            # F0 contour
            axes[2].plot(features['times'], features['f0'], label='F0', linewidth=1.5)
            axes[2].set_ylabel('Frequency (Hz)')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_title('F0 Contour')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {output_file}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            print("matplotlib not available for visualization")


if __name__ == "__main__":
    # Test melody extractor
    extractor = MelodyExtractor()

    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(test_file).exists():
        print(f"Testing with {test_file}")

        # Extract features
        features = extractor.extract_melody_contour(test_file)

        print(f"Chroma shape: {features['chroma'].shape}")
        print(f"CQT top-4 shape: {features['cqt_top4'].shape}")
        print(f"F0 shape: {features['f0'].shape}")

        # Save features
        extractor.save_melody_features(test_file, "./outputs/melody_features")
