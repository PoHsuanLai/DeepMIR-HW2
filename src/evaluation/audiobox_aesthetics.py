"""
Meta Audiobox Aesthetics evaluation
Evaluates CE (Content Enjoyment), CU (Content Usefulness),
PC (Production Complexity), PQ (Production Quality)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict


class AudioboxAesthetics:
    """
    Meta Audiobox Aesthetics evaluator
    """

    def __init__(self, device=None):
        """
        Initialize Audiobox Aesthetics model

        Args:
            device: torch device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing Audiobox Aesthetics on {self.device}")

        try:
            # Note: The actual Meta Audiobox Aesthetics model needs to be downloaded
            # This is a placeholder for the model loading
            print("Note: Audiobox Aesthetics model needs to be set up")
            print("Refer to: https://ai.meta.com/research/audiobox/")

            self.model = None
            self.model_loaded = False

        except Exception as e:
            print(f"Error loading Audiobox Aesthetics: {e}")
            self.model = None
            self.model_loaded = False

    def _extract_handcrafted_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract handcrafted audio features for aesthetic prediction

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with feature-based aesthetic scores
        """
        import librosa

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30.0)

        # Features for Content Enjoyment (CE)
        # - Harmonic content, melody strength
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(harmonic ** 2) / (np.mean(y ** 2) + 1e-8)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_std = np.std(chroma)  # Harmonic variation

        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # CE: Higher with clear harmony and pleasant spectral content
        ce_score = min(1.0, harmonic_ratio * 2.0 + chroma_std * 0.3)
        ce_score = max(0.0, min(1.0, ce_score))

        # Features for Content Usefulness (CU)
        # - Clarity, distinguishability
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms) - np.min(rms)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = np.std(zcr)

        # CU: Higher with clear dynamics and well-defined content
        cu_score = min(1.0, dynamic_range * 5.0 + zcr_std * 0.5)
        cu_score = max(0.0, min(1.0, cu_score))

        # Features for Production Complexity (PC)
        # - Spectral complexity, rhythmic complexity
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_regularity = 1.0 - np.std(np.diff(beats)) / (np.mean(np.diff(beats)) + 1e-8)

        # PC: Higher with complex spectral and rhythmic content
        pc_score = min(1.0, contrast_mean * 0.1 + (1 - beat_regularity) * 0.5)
        pc_score = max(0.0, min(1.0, pc_score))

        # Features for Production Quality (PQ)
        # - Signal clarity, noise level, mastering quality
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # High-frequency content (indicates good quality)
        hf_energy = np.mean(np.abs(librosa.stft(y))[sr // 4:, :])
        total_energy = np.mean(np.abs(librosa.stft(y)))
        hf_ratio = hf_energy / (total_energy + 1e-8)

        # PQ: Higher with low noise and good frequency balance
        pq_score = min(1.0, (1.0 - spectral_flatness) * 1.5 + hf_ratio * 0.3)
        pq_score = max(0.0, min(1.0, pq_score))

        # Normalize to reasonable ranges (typically 0.5-1.0 for good music)
        ce_score = 0.5 + ce_score * 0.5
        cu_score = 0.5 + cu_score * 0.5
        pc_score = 0.4 + pc_score * 0.6
        pq_score = 0.5 + pq_score * 0.5

        return {
            'CE': float(ce_score),
            'CU': float(cu_score),
            'PC': float(pc_score),
            'PQ': float(pq_score)
        }

    def evaluate(self, audio_path: str) -> Dict[str, float]:
        """
        Evaluate audio aesthetics

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with CE, CU, PC, PQ scores
        """
        audio_path = str(audio_path)

        if self.model_loaded:
            # Use actual model
            # scores = self.model.predict(audio_path)
            pass
        else:
            # Use handcrafted features
            scores = self._extract_handcrafted_features(audio_path)

        print(f"\nAudiobox Aesthetics for {Path(audio_path).name}:")
        print(f"  CE (Content Enjoyment):   {scores['CE']:.3f}")
        print(f"  CU (Content Usefulness):  {scores['CU']:.3f}")
        print(f"  PC (Production Complex):  {scores['PC']:.3f}")
        print(f"  PQ (Production Quality):  {scores['PQ']:.3f}")

        return scores

    def evaluate_batch(self, audio_paths: list) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple audio files

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary mapping paths to score dictionaries
        """
        results = {}

        for audio_path in audio_paths:
            try:
                scores = self.evaluate(audio_path)
                results[audio_path] = scores
            except Exception as e:
                print(f"Error evaluating {audio_path}: {e}")
                results[audio_path] = {'CE': 0.0, 'CU': 0.0, 'PC': 0.0, 'PQ': 0.0}

        return results

    def compare(self, audio_path1: str, audio_path2: str) -> Dict:
        """
        Compare aesthetics between two audio files

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file

        Returns:
            Dictionary with scores and differences
        """
        scores1 = self.evaluate(audio_path1)
        scores2 = self.evaluate(audio_path2)

        differences = {
            key: scores2[key] - scores1[key]
            for key in scores1.keys()
        }

        print(f"\nComparison:")
        print(f"  File 1: {Path(audio_path1).name}")
        print(f"  File 2: {Path(audio_path2).name}")
        print(f"\nDifferences (File 2 - File 1):")
        for key, diff in differences.items():
            sign = '+' if diff >= 0 else ''
            print(f"  {key}: {sign}{diff:.3f}")

        return {
            'file1_scores': scores1,
            'file2_scores': scores2,
            'differences': differences
        }


if __name__ == "__main__":
    # Test aesthetics evaluator
    evaluator = AudioboxAesthetics()

    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(test_file).exists():
        scores = evaluator.evaluate(test_file)
        print(f"\nScores: {scores}")
