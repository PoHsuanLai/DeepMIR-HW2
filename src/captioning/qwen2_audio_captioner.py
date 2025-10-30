"""
Audio captioning using Qwen2-Audio
Supports text prompts for targeted caption generation
"""

import torch
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')


class Qwen2AudioCaptioner:
    """
    Audio captioning using Qwen2-Audio (7B instruct model)
    Supports custom text prompts for guided caption generation
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-Audio-7B-Instruct", device: Optional[str] = None):
        """
        Initialize Qwen2-Audio captioner

        Args:
            model_name: Model to use (default: Qwen/Qwen2-Audio-7B-Instruct)
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        print(f"Initializing Qwen2AudioCaptioner with {model_name} on {self.device}")
        print("This may take a few minutes to download the model...")

        self._init_qwen2_audio()

    def _init_qwen2_audio(self):
        """Initialize Qwen2-Audio model"""
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

            print("Loading Qwen2-Audio model...")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with automatic device mapping
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
            )

            if self.device != 'cuda':
                self.model = self.model.to(self.device)

            self.model.eval()

            print("Qwen2-Audio loaded successfully!")

        except Exception as e:
            print(f"Error loading Qwen2-Audio: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Qwen2-Audio: {e}")

    def caption_audio(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        target_duration: Optional[float] = None,
        target_total_length: Optional[int] = None,
    ) -> str:
        """
        Generate caption for audio file with optional text prompt

        Args:
            audio_path: Path to audio file
            prompt: Text prompt to guide caption generation
                   Examples:
                   - "Describe the instruments, tempo, and mood"
                   - "What is the genre and style of this music?"
                   - "Describe the drum pattern and rhythm"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more creative)
            target_duration: Duration to use from audio (in seconds)
            target_total_length: If provided, will truncate caption to this length

        Returns:
            Generated caption string
        """
        if prompt is None:
            # Default prompt for music captioning
            prompt = "Describe this audio in detail, including the instruments, musical style, tempo, mood, and any notable characteristics."

        audio_path = str(audio_path)

        print(f"Captioning: {Path(audio_path).name}")
        print(f"Prompt: {prompt}")

        try:
            import librosa

            # Load audio and resample to 16kHz (required by Qwen2-Audio)
            target_sr = 16000
            if target_duration:
                audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=target_duration)
            else:
                audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

            # Prepare conversation format (audio analysis mode)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            # Process inputs
            inputs = self.processor(
                text=text,
                audios=[audio],
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = inputs.to(self.model.device)

            # Generate
            with torch.inference_mode():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                )

            # Remove input tokens from generation
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            # Decode
            caption = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Truncate if needed
            if target_total_length and len(caption) > target_total_length:
                # Truncate at sentence boundary
                caption = caption[:target_total_length]
                if '.' in caption:
                    caption = caption.rsplit('.', 1)[0] + '.'

            print(f"Generated caption ({len(caption)} chars): {caption[:100]}...")

            return caption

        except Exception as e:
            print(f"Error during caption generation: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating caption"

    def caption_batch(self, audio_paths: list, prompt: Optional[str] = None) -> dict:
        """
        Generate captions for multiple audio files

        Args:
            audio_paths: List of audio file paths
            prompt: Text prompt to use for all files

        Returns:
            Dictionary mapping paths to captions
        """
        captions = {}

        for audio_path in audio_paths:
            try:
                caption = self.caption_audio(audio_path, prompt=prompt)
                captions[audio_path] = caption
                print(f"\n{Path(audio_path).name}:")
                print(f"  {caption}")
            except Exception as e:
                print(f"Error captioning {audio_path}: {e}")
                captions[audio_path] = "Error generating caption"

        return captions


if __name__ == "__main__":
    # Test captioner
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    args = parser.parse_args()

    captioner = Qwen2AudioCaptioner()
    caption = captioner.caption_audio(
        args.audio_file,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )

    print("\n" + "=" * 80)
    print("FINAL CAPTION:")
    print("=" * 80)
    print(caption)
