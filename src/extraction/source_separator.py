"""
Source Separation using Demucs for Enhanced Music Conditioning
Separates audio into vocals, drums, bass, and other instruments
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tempfile
import os


class SourceSeparator:
    """
    Extract separated audio stems using Demucs (Meta's SOTA model)
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cuda"):
        """
        Initialize Demucs source separator

        Args:
            model_name: Demucs model variant (htdemucs, htdemucs_ft, mdx_extra)
                       htdemucs = Hybrid Transformer Demucs (recommended)
            device: Device for processing (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.sample_rate = 44100

        print(f"Initializing Demucs ({model_name}) on {device}")
        self._load_model()

    def _load_model(self):
        """Load Demucs model from HuggingFace or torchhub"""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            # Load pretrained Demucs model
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Store apply function for inference
            self.apply_model = apply_model

            print(f"Demucs model loaded successfully")
            print(f"  Model: {self.model_name}")
            print(f"  Sources: {self.model.sources}")  # ['drums', 'bass', 'other', 'vocals']

        except Exception as e:
            print(f"Error loading Demucs: {e}")
            raise RuntimeError(f"Failed to load Demucs model: {e}")

    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems

        Args:
            audio_path: Path to input audio file
            output_dir: Optional directory to save separated stems

        Returns:
            Dictionary with separated stems: {stem_name: audio_array}
            Keys: 'drums', 'bass', 'other', 'vocals'
        """
        print(f"Separating audio: {audio_path}")

        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
            wav = resampler(wav)
            sr = self.model.samplerate

        # Convert to mono if stereo (Demucs handles stereo, but for consistency)
        if wav.shape[0] > 1:
            # Keep stereo for better separation quality
            pass
        else:
            # Duplicate mono to stereo
            wav = wav.repeat(2, 1)

        # Move to device and add batch dimension
        wav = wav.to(self.device).unsqueeze(0)

        # Apply Demucs separation
        with torch.no_grad():
            sources = self.apply_model(
                self.model,
                wav,
                device=self.device,
                split=True,  # Split into chunks for memory efficiency
                overlap=0.25
            )

        # Convert to dictionary: {source_name: numpy_array}
        separated_stems = {}
        for i, source_name in enumerate(self.model.sources):
            # Extract source (batch, source, channel, time)
            stem = sources[0, i].cpu().numpy()  # (channels, samples)

            # Convert to mono by averaging channels
            if stem.shape[0] > 1:
                stem = stem.mean(axis=0)
            else:
                stem = stem[0]

            separated_stems[source_name] = stem
            print(f"  {source_name}: {stem.shape}")

        # Save stems if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = Path(audio_path).stem
            for stem_name, stem_audio in separated_stems.items():
                output_path = output_dir / f"{base_name}_{stem_name}.wav"
                torchaudio.save(
                    str(output_path),
                    torch.from_numpy(stem_audio).unsqueeze(0),
                    self.model.samplerate
                )
                print(f"  Saved: {output_path}")

        return separated_stems

    def get_instrumental(self, audio_path: str) -> np.ndarray:
        """
        Get instrumental mix (no vocals)

        Args:
            audio_path: Path to input audio

        Returns:
            Instrumental audio (drums + bass + other)
        """
        stems = self.separate(audio_path)

        # Mix everything except vocals
        instrumental = (
            stems['drums'] +
            stems['bass'] +
            stems['other']
        )

        # Normalize to prevent clipping
        max_val = np.abs(instrumental).max()
        if max_val > 0:
            instrumental = instrumental / max_val * 0.9

        return instrumental

    def get_drums_only(self, audio_path: str) -> np.ndarray:
        """Extract only drums for rhythm conditioning"""
        stems = self.separate(audio_path)
        return stems['drums']

    def get_melodic_instruments(self, audio_path: str) -> np.ndarray:
        """
        Get melodic instruments (no drums, no vocals)
        Best for melody extraction
        """
        stems = self.separate(audio_path)

        # Bass + Other (melodic instruments)
        melodic = stems['bass'] + stems['other']

        # Normalize
        max_val = np.abs(melodic).max()
        if max_val > 0:
            melodic = melodic / max_val * 0.9

        return melodic

    def save_stem_as_tempfile(self, stem_audio: np.ndarray, sr: int = 44100) -> str:
        """
        Save stem to temporary file for use in models

        Args:
            stem_audio: Audio array
            sr: Sample rate

        Returns:
            Path to temporary file
        """
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Save audio
        torchaudio.save(
            temp_path,
            torch.from_numpy(stem_audio).unsqueeze(0),
            sr
        )

        return temp_path
