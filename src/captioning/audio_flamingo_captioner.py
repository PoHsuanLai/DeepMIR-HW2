"""
Audio captioning using NVIDIA Audio Flamingo 3
Much more powerful than LP-MusicCaps, handles speech, sound, and music
"""

import torch
import sys
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AudioFlamingoCaptioner:
    """
    Audio captioning using Audio Flamingo 3 (7B model)
    """

    def __init__(self, model_name: str = "nvidia/audio-flamingo-3", device: Optional[str] = None):
        """
        Initialize Audio Flamingo 3 captioner

        Args:
            model_name: Model to use (default: nvidia/audio-flamingo-3)
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        print(f"Initializing AudioFlamingoCaptioner with {model_name} on {self.device}")

        # Add Audio Flamingo to path
        af_path = Path(__file__).parent.parent.parent / "audio_flamingo" / "audio-flamingo"
        if af_path.exists():
            sys.path.insert(0, str(af_path))
        else:
            raise RuntimeError(f"Audio Flamingo not found at {af_path}. Please install it first.")

        self._init_audio_flamingo()

    def _init_audio_flamingo(self):
        """Initialize Audio Flamingo 3 model"""
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import process_anyvalue, tokenizer_audio_token
            from llava.constants import AUDIO_TOKEN_INDEX
            from llava.conversation import conv_templates

            print("Loading Audio Flamingo 3 model (this may take a few minutes)...")

            # Load model
            self.tokenizer, self.model, self.context_len = load_pretrained_model(
                model_path=self.model_name,
                model_base=None,
                model_name=self.model_name,
                device=self.device
            )

            self.model.eval()

            # Store utilities
            self.process_anyvalue = process_anyvalue
            self.tokenizer_audio_token = tokenizer_audio_token
            self.AUDIO_TOKEN_INDEX = AUDIO_TOKEN_INDEX
            self.conv_templates = conv_templates

            print("Audio Flamingo 3 loaded successfully!")

        except Exception as e:
            print(f"Error loading Audio Flamingo 3: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Audio Flamingo 3: {e}")

    def caption_audio(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        target_duration: Optional[float] = None,  # For compatibility
        target_total_length: Optional[int] = None,  # For compatibility
    ) -> str:
        """
        Generate caption for audio file

        Args:
            audio_path: Path to audio file
            prompt: Custom prompt (default: "Please describe this audio in detail")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            target_duration: Ignored (for compatibility with LP-MusicCaps interface)
            target_total_length: If provided, will truncate caption to this length

        Returns:
            Generated caption string
        """
        if prompt is None:
            prompt = "Please describe this audio in detail, including instruments, mood, tempo, and style."

        audio_path = str(audio_path)

        print(f"Captioning: {Path(audio_path).name}")
        print(f"Prompt: {prompt}")

        try:
            # Prepare conversation
            conv_mode = "auto"  # Auto-detect conversation mode
            conv = self.conv_templates[conv_mode].copy()

            # Add audio token to prompt
            inp = f"<sound>\n{prompt}"
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_with_template = conv.get_prompt()

            # Tokenize
            input_ids = self.tokenizer_audio_token(
                prompt_with_template,
                self.tokenizer,
                self.AUDIO_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)

            # Process audio
            audio_tensors = self.process_anyvalue(
                audio_path,
                self.model.get_model().get_audio_tower(),
                None  # No video
            ).to(dtype=self.model.dtype, device=self.device)

            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    audio=audio_tensors,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            # Decode
            caption = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )[0].strip()

            # Remove the prompt from the output if it's included
            if prompt in caption:
                caption = caption.split(prompt)[-1].strip()

            # Truncate if needed
            if target_total_length and len(caption) > target_total_length:
                caption = caption[:target_total_length].rsplit('.', 1)[0] + '.'

            print(f"Generated caption ({len(caption)} chars): {caption[:100]}...")

            return caption

        except Exception as e:
            print(f"Error during caption generation: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating caption"

    def caption_batch(self, audio_paths: list) -> dict:
        """
        Generate captions for multiple audio files

        Args:
            audio_paths: List of audio file paths

        Returns:
            Dictionary mapping paths to captions
        """
        captions = {}

        for audio_path in audio_paths:
            try:
                caption = self.caption_audio(audio_path)
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
    args = parser.parse_args()

    captioner = AudioFlamingoCaptioner()
    caption = captioner.caption_audio(args.audio_file, prompt=args.prompt)

    print("\n" + "=" * 80)
    print("FINAL CAPTION:")
    print("=" * 80)
    print(caption)
