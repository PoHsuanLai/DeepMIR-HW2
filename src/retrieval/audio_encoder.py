"""
Audio Encoder for Music Retrieval
Uses CLAP (Contrastive Language-Audio Pretraining) for encoding audio into embeddings
"""

import torch
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from typing import Union, List


class AudioEncoder:
    """
    Audio encoder using CLAP model for extracting audio embeddings
    """

    def __init__(self, model_name='laion/larger_clap_music_and_speech', device=None):
        """
        Initialize CLAP audio encoder via Hugging Face

        Args:
            model_name: CLAP model from Hugging Face Hub
            device: torch device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing AudioEncoder on {self.device}")

        from transformers import ClapModel, ClapProcessor

        print(f"Loading CLAP from Hugging Face: {model_name}")
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)

        print(f"CLAP model loaded successfully")

    def preprocess_audio(self, audio_path: str, target_sr: int = 48000, duration: float = None):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            duration: Duration to load (None for full file)

        Returns:
            Preprocessed audio array
        """
        audio_path = str(audio_path)

        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)

        # Convert to mono if needed
        if len(y.shape) > 1:
            y = librosa.to_mono(y)

        return y, sr

    def encode_audio(self, audio_path: Union[str, Path], normalize: bool = True) -> np.ndarray:
        """
        Encode single audio file to embedding using Hugging Face CLAP

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize the embedding

        Returns:
            Audio embedding as numpy array
        """
        audio_path = str(audio_path)

        # Load and preprocess audio
        audio_data, sr = librosa.load(audio_path, sr=48000, mono=True)

        # Process with CLAP (transformers 4.46.0 uses 'audios' parameter)
        inputs = self.processor(audios=audio_data, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**inputs)

        # Convert to numpy
        audio_embed = audio_embed.cpu().numpy()

        if normalize:
            audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=1, keepdims=True)

        return audio_embed[0]


    def encode_batch(self, audio_paths: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode multiple audio files in batch

        Args:
            audio_paths: List of audio file paths
            normalize: Whether to normalize embeddings

        Returns:
            Array of embeddings, shape (n_files, embedding_dim)
        """
        embeddings = []

        for audio_path in audio_paths:
            try:
                embed = self.encode_audio(audio_path, normalize=normalize)
                embeddings.append(embed)
            except Exception as e:
                print(f"Error encoding {audio_path}: {e}")
                continue

        return np.array(embeddings)

    def encode_directory(self, directory: Union[str, Path],
                        extensions: tuple = ('.wav', '.mp3', '.flac', '.m4a'),
                        normalize: bool = True) -> dict:
        """
        Encode all audio files in a directory

        Args:
            directory: Path to directory
            extensions: Audio file extensions to process
            normalize: Whether to normalize embeddings

        Returns:
            Dictionary mapping file paths to embeddings
        """
        directory = Path(directory)
        audio_files = []

        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))

        print(f"Found {len(audio_files)} audio files in {directory}")

        embeddings_dict = {}
        for audio_file in audio_files:
            try:
                embed = self.encode_audio(audio_file, normalize=normalize)
                embeddings_dict[str(audio_file)] = embed
                print(f"Encoded: {audio_file.name}")
            except Exception as e:
                print(f"Error encoding {audio_file}: {e}")

        return embeddings_dict


if __name__ == "__main__":
    # Test the encoder
    encoder = AudioEncoder()

    # Test on a single file
    test_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"
    if Path(test_file).exists():
        embedding = encoder.encode_audio(test_file)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10 values): {embedding[:10]}")
