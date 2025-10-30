"""
CLAP-based similarity evaluation
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity


class CLAPSimilarity:
    """
    CLAP-based similarity calculator for music evaluation
    """

    def __init__(self, device=None):
        """
        Initialize CLAP model for similarity calculation via Hugging Face

        Args:
            device: torch device (cuda/cpu)
        """
        import torch
        import librosa
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing CLAP for similarity on {self.device}")

        from transformers import ClapModel, ClapProcessor

        model_name = "laion/larger_clap_music_and_speech"
        print(f"Loading CLAP from Hugging Face: {model_name}")
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)

        print("CLAP model loaded successfully")

    def encode_audio(self, audio_path: str) -> np.ndarray:
        """
        Encode audio file to embedding using Hugging Face CLAP

        Args:
            audio_path: Path to audio file

        Returns:
            Audio embedding
        """
        import librosa
        import torch

        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=48000, mono=True)

        # Process with CLAP
        inputs = self.processor(audios=audio_data, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            audio_embed = self.model.get_audio_features(**inputs)

        # Convert to numpy and normalize
        audio_embed = audio_embed.cpu().numpy()
        audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=1, keepdims=True)

        return audio_embed[0]

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding using Hugging Face CLAP

        Args:
            text: Text description

        Returns:
            Text embedding
        """
        import torch

        # Process text with CLAP
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            text_embed = self.model.get_text_features(**inputs)

        # Convert to numpy and normalize
        text_embed = text_embed.cpu().numpy()
        text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)

        return text_embed[0]

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        # Reshape if needed
        if len(embedding1.shape) == 1:
            embedding1 = embedding1.reshape(1, -1)
        if len(embedding2.shape) == 1:
            embedding2 = embedding2.reshape(1, -1)

        similarity = cosine_similarity(embedding1, embedding2)[0, 0]
        return float(similarity)

    def audio_to_audio_similarity(self, audio_path1: str, audio_path2: str) -> float:
        """
        Calculate similarity between two audio files

        Args:
            audio_path1: First audio file
            audio_path2: Second audio file

        Returns:
            Similarity score
        """
        embed1 = self.encode_audio(audio_path1)
        embed2 = self.encode_audio(audio_path2)

        return self.calculate_similarity(embed1, embed2)

    def text_to_audio_similarity(self, text: str, audio_path: str) -> float:
        """
        Calculate similarity between text and audio

        Args:
            text: Text description
            audio_path: Audio file path

        Returns:
            Similarity score
        """
        text_embed = self.encode_text(text)
        audio_embed = self.encode_audio(audio_path)

        return self.calculate_similarity(text_embed, audio_embed)

    def evaluate_generation(self,
                           target_audio: str,
                           generated_audio: str,
                           text_prompt: str) -> Dict[str, float]:
        """
        Complete CLAP evaluation for generation task

        Args:
            target_audio: Target audio file path
            generated_audio: Generated audio file path
            text_prompt: Text prompt used for generation

        Returns:
            Dictionary with similarity scores
        """
        print(f"\nEvaluating CLAP similarity:")
        print(f"  Target: {Path(target_audio).name}")
        print(f"  Generated: {Path(generated_audio).name}")
        print(f"  Text: {text_prompt[:60]}...")

        # 1. Target music <-> Input text
        target_text_sim = self.text_to_audio_similarity(text_prompt, target_audio)

        # 2. Input text <-> Generated music
        text_generated_sim = self.text_to_audio_similarity(text_prompt, generated_audio)

        # 3. Generated music <-> Target music
        generated_target_sim = self.audio_to_audio_similarity(generated_audio, target_audio)

        results = {
            'target_text_similarity': target_text_sim,
            'text_generated_similarity': text_generated_sim,
            'generated_target_similarity': generated_target_sim
        }

        print(f"\nResults:")
        print(f"  Target ↔ Text: {target_text_sim:.4f}")
        print(f"  Text ↔ Generated: {text_generated_sim:.4f}")
        print(f"  Generated ↔ Target: {generated_target_sim:.4f}")

        return results

    def evaluate_retrieval(self,
                          target_audio: str,
                          retrieved_audio: str) -> float:
        """
        Evaluate retrieval quality

        Args:
            target_audio: Target audio file path
            retrieved_audio: Retrieved audio file path

        Returns:
            Similarity score
        """
        similarity = self.audio_to_audio_similarity(target_audio, retrieved_audio)

        print(f"\nRetrieval CLAP similarity:")
        print(f"  Target: {Path(target_audio).name}")
        print(f"  Retrieved: {Path(retrieved_audio).name}")
        print(f"  Similarity: {similarity:.4f}")

        return similarity


if __name__ == "__main__":
    # Test CLAP similarity
    evaluator = CLAPSimilarity()

    # Test files
    target_file = "./fundwotsai/Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"

    if Path(target_file).exists():
        # Test text-to-audio similarity
        text = "A jazz music piece with drums and bass, upbeat and energetic"
        similarity = evaluator.text_to_audio_similarity(text, target_file)
        print(f"Text-to-audio similarity: {similarity:.4f}")

        # Test audio-to-audio similarity (with itself)
        self_sim = evaluator.audio_to_audio_similarity(target_file, target_file)
        print(f"Self-similarity: {self_sim:.4f}")
