"""
Factory for creating caption models with unified interface
"""

from typing import Optional, Union
from pathlib import Path


class UnifiedCaptioner:
    """
    Unified interface for all caption models
    """

    def __init__(self, captioner_instance, model_type: str, prompt: Optional[str] = None):
        self.captioner = captioner_instance
        self.model_type = model_type
        self.default_prompt = prompt

    def caption_audio(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
        target_duration: Optional[float] = None,
        target_total_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate caption with unified interface

        Args:
            audio_path: Path to audio file
            prompt: Text prompt (for prompted models like Qwen2-Audio)
            target_duration: Duration to caption in seconds
            target_total_length: Target caption length in characters
            **kwargs: Additional model-specific parameters

        Returns:
            Generated caption string
        """
        if self.model_type == 'qwen2-audio':
            # Qwen2-Audio: prompted model
            # Note: Don't pass target_total_length - let the model generate naturally
            # MusicGen's 256-token limit is for tokens, not characters
            # A 400-500 char caption still fits within token budget
            actual_prompt = prompt or self.default_prompt
            return self.captioner.caption_audio(
                audio_path,
                prompt=actual_prompt,
                target_duration=target_duration,
                target_total_length=None,  # Let model generate full caption
                temperature=kwargs.get('temperature', 0.7),
                max_new_tokens=kwargs.get('max_new_tokens', 256)
            )
        else:
            # LP-MusicCaps: non-prompted model
            return self.captioner.caption_audio(
                audio_path,
                target_duration=target_duration,
                target_total_length=target_total_length
            )


def create_captioner(model_name: str, config: dict = None) -> UnifiedCaptioner:
    """
    Factory function to create caption models

    Args:
        model_name: Name of caption model ('qwen2-audio' or 'lpmusiccaps')
        config: Optional configuration dict

    Returns:
        UnifiedCaptioner instance
    """
    config = config or {}

    if model_name == 'qwen2-audio':
        from captioning.qwen2_audio_captioner import Qwen2AudioCaptioner

        print("📝 Initializing Caption Model: Qwen2-Audio-7B (prompted model)")
        print("   This may take a few minutes to download (~15GB)...")

        captioner = Qwen2AudioCaptioner(
            model_name="Qwen/Qwen2-Audio-7B-Instruct"
        )

        # Optimized prompt for better music generation results
        default_prompt = (
            "Describe this music in detail. Include: "
            "1) The instruments being played (e.g., drums, bass, guitar, piano, vocals), "
            "2) The musical genre and style (e.g., jazz, rock, electronic, classical), "
            "3) The tempo and rhythm (e.g., fast, slow, steady beat), "
            "4) The mood and atmosphere (e.g., energetic, calm, mysterious, upbeat), "
            "5) Any notable musical characteristics (e.g., harmony, melody patterns, dynamics)."
        )

        print("✅ Qwen2-Audio loaded successfully!")

        return UnifiedCaptioner(captioner, 'qwen2-audio', prompt=default_prompt)

    elif model_name == 'lpmusiccaps':
        from captioning.audio_captioner import AudioCaptioner

        print("📝 Initializing Caption Model: LP-MusicCaps")

        max_caption_length = config.get('max_caption_length', 128)

        captioner = AudioCaptioner(
            model_name='lpmusiccaps',
            max_caption_length=max_caption_length
        )

        print("✅ LP-MusicCaps loaded successfully!")

        return UnifiedCaptioner(captioner, 'lpmusiccaps', prompt=None)

    elif model_name == 'qwen2-audio-natural':
        from captioning.qwen2_audio_captioner import Qwen2AudioCaptioner

        print("📝 Initializing Caption Model: Qwen2-Audio-7B (natural language mode)")
        print("   Optimized for CLAP similarity scoring")

        captioner = Qwen2AudioCaptioner(
            model_name="Qwen/Qwen2-Audio-7B-Instruct"
        )

        # CLAP-optimized prompt focused on controllable aspects (melody/chords/drums)
        # Avoids specific instruments since JASCO can't distinguish them
        default_prompt = (
            "Describe the musical characteristics of this piece in simple, natural language. "
            "Focus on: "
            "1) The melody - is it flowing and smooth, or jumpy and energetic? High or low? "
            "2) The rhythm and beat - is it steady or varied? Fast or slow? "
            "3) The harmonic progression - does it feel bright and major, or dark and minor? "
            "4) The overall mood and energy level. "
            "Do NOT mention specific instruments (like piano, flute, guitar). "
            "Instead describe the melodic and rhythmic qualities. "
            "Avoid technical terms like BPM, time signatures, or key names. "
            "Write as if describing the musical feel to a friend."
        )

        print("✅ Qwen2-Audio loaded successfully (natural language mode)!")

        return UnifiedCaptioner(captioner, 'qwen2-audio', prompt=default_prompt)

    else:
        raise ValueError(f"Unknown caption model: {model_name}. Use 'qwen2-audio', 'qwen2-audio-natural', or 'lpmusiccaps'")
