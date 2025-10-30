"""
Melody Similarity/Accuracy evaluation
Compares melody content between generated and target audio
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score


class MelodyAccuracy:
    """
    Melody accuracy evaluator using various methods
    """

    def __init__(self, sr: int = 22050):
        """
        Initialize melody accuracy evaluator

        Args:
            sr: Sample rate for processing
        """
        self.sr = sr

    def load_and_trim(self, audio_path: str, duration: float) -> Tuple[np.ndarray, int]:
        """
        Load and trim audio to specified duration

        Args:
            audio_path: Path to audio file
            duration: Duration in seconds

        Returns:
            Audio array and sample rate
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
        return y, sr

    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract chromagram

        Args:
            y: Audio array
            sr: Sample rate

        Returns:
            Chromagram
        """
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_chroma=12,
            hop_length=512
        )
        return chroma

    def extract_cqt(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract CQT

        Args:
            y: Audio array
            sr: Sample rate

        Returns:
            CQT magnitude
        """
        cqt = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84)
        return np.abs(cqt)

    def chroma_similarity(self, audio_path1: str, audio_path2: str,
                         duration: float = None) -> float:
        """
        Calculate melody similarity using chromagram

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file
            duration: Duration to compare (None for full)

        Returns:
            Similarity score (0-1)
        """
        # Load audio
        y1, sr1 = self.load_and_trim(audio_path1, duration)
        y2, sr2 = self.load_and_trim(audio_path2, duration)

        # Extract chroma
        chroma1 = self.extract_chroma(y1, sr1)
        chroma2 = self.extract_chroma(y2, sr2)

        # Align to same length
        min_len = min(chroma1.shape[1], chroma2.shape[1])
        chroma1 = chroma1[:, :min_len]
        chroma2 = chroma2[:, :min_len]

        # Calculate cosine similarity
        chroma1_flat = chroma1.flatten()
        chroma2_flat = chroma2.flatten()

        similarity = 1.0 - cosine(chroma1_flat, chroma2_flat)

        return float(max(0.0, min(1.0, similarity)))

    def chroma_accuracy(self, audio_path1: str, audio_path2: str,
                       duration: float = None, threshold: float = 0.5) -> float:
        """
        Calculate melody accuracy by comparing dominant pitch classes

        Args:
            audio_path1: First audio file (target)
            audio_path2: Second audio file (generated)
            duration: Duration to compare
            threshold: Threshold for pitch class activation

        Returns:
            Accuracy score (0-1)
        """
        # Load audio
        y1, sr1 = self.load_and_trim(audio_path1, duration)
        y2, sr2 = self.load_and_trim(audio_path2, duration)

        # Extract chroma
        chroma1 = self.extract_chroma(y1, sr1)
        chroma2 = self.extract_chroma(y2, sr2)

        # Align to same length
        min_len = min(chroma1.shape[1], chroma2.shape[1])
        chroma1 = chroma1[:, :min_len]
        chroma2 = chroma2[:, :min_len]

        # Get dominant pitch class for each frame
        dominant1 = np.argmax(chroma1, axis=0)
        dominant2 = np.argmax(chroma2, axis=0)

        # Calculate accuracy
        accuracy = accuracy_score(dominant1, dominant2)

        return float(accuracy)

    def pitch_contour_similarity(self, audio_path1: str, audio_path2: str,
                                duration: float = None) -> float:
        """
        Calculate melody similarity using F0 contours

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file
            duration: Duration to compare

        Returns:
            Similarity score (0-1)
        """
        # Load audio
        y1, sr1 = self.load_and_trim(audio_path1, duration)
        y2, sr2 = self.load_and_trim(audio_path2, duration)

        # Extract F0
        try:
            f0_1, _, _ = librosa.pyin(y1, fmin=65, fmax=2093, sr=sr1)
            f0_2, _, _ = librosa.pyin(y2, fmin=65, fmax=2093, sr=sr2)
        except:
            # Fallback to chroma if F0 extraction fails
            return self.chroma_similarity(audio_path1, audio_path2, duration)

        # Handle NaN values
        f0_1 = np.nan_to_num(f0_1, nan=0.0)
        f0_2 = np.nan_to_num(f0_2, nan=0.0)

        # Align lengths
        min_len = min(len(f0_1), len(f0_2))
        f0_1 = f0_1[:min_len]
        f0_2 = f0_2[:min_len]

        # Normalize to pitch classes (ignore octave)
        def freq_to_pitch_class(f0):
            pitch_class = np.zeros_like(f0)
            mask = f0 > 0
            pitch_class[mask] = 12 * np.log2(f0[mask] / 440.0) % 12
            return pitch_class

        pc1 = freq_to_pitch_class(f0_1)
        pc2 = freq_to_pitch_class(f0_2)

        # Calculate correlation
        valid_mask = (f0_1 > 0) & (f0_2 > 0)
        if np.sum(valid_mask) > 10:
            correlation = np.corrcoef(pc1[valid_mask], pc2[valid_mask])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Convert to 0-1 range
        similarity = (correlation + 1.0) / 2.0

        return float(max(0.0, min(1.0, similarity)))

    def melody_accuracy_comprehensive(self, target_path: str, generated_path: str,
                                     duration: float = None) -> Dict[str, float]:
        """
        Comprehensive melody accuracy evaluation

        Args:
            target_path: Target audio file path
            generated_path: Generated audio file path
            duration: Duration to compare (should match model output length)

        Returns:
            Dictionary with multiple accuracy metrics
        """
        print(f"\nEvaluating melody accuracy:")
        print(f"  Target: {Path(target_path).name}")
        print(f"  Generated: {Path(generated_path).name}")
        if duration:
            print(f"  Duration: {duration:.1f}s")

        # Chroma similarity
        chroma_sim = self.chroma_similarity(target_path, generated_path, duration)

        # Chroma accuracy (dominant pitch class)
        chroma_acc = self.chroma_accuracy(target_path, generated_path, duration)

        # Pitch contour similarity
        pitch_sim = self.pitch_contour_similarity(target_path, generated_path, duration)

        # Overall melody accuracy (weighted average)
        overall_accuracy = (chroma_sim * 0.3 + chroma_acc * 0.4 + pitch_sim * 0.3)

        results = {
            'chroma_similarity': chroma_sim,
            'chroma_accuracy': chroma_acc,
            'pitch_contour_similarity': pitch_sim,
            'overall_melody_accuracy': overall_accuracy
        }

        print(f"\nResults:")
        print(f"  Chroma Similarity: {chroma_sim:.4f}")
        print(f"  Chroma Accuracy: {chroma_acc:.4f}")
        print(f"  Pitch Contour Similarity: {pitch_sim:.4f}")
        print(f"  Overall Melody Accuracy: {overall_accuracy:.4f}")

        return results

    def evaluate(self, target_path: str, generated_path: str,
                duration: float = None) -> float:
        """
        Simple evaluation returning single accuracy score

        Args:
            target_path: Target audio file path
            generated_path: Generated audio file path
            duration: Duration to compare

        Returns:
            Single accuracy score
        """
        results = self.melody_accuracy_comprehensive(target_path, generated_path, duration)
        return results['overall_melody_accuracy']


if __name__ == "__main__":
    # Test melody accuracy
    evaluator = MelodyAccuracy()

    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(test_file).exists():
        # Test with itself (should have high similarity)
        results = evaluator.melody_accuracy_comprehensive(
            test_file,
            test_file,
            duration=30.0
        )

        print(f"\nSelf-comparison results (should be ~1.0):")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
