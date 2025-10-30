"""
Audio Captioning using LP-MusicCaps model
Uses seungheondoh/lp-music-caps for detailed music captioning
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Union


class AudioCaptioner:
    """
    Audio captioning using LP-MusicCaps
    """

    def __init__(self, model_name: str = "lpmusiccaps", device: Optional[str] = None, max_caption_length: int = 128):
        """
        Initialize audio captioner

        Args:
            model_name: Model to use ('lpmusiccaps')
            device: Device to use (cuda/cpu)
            max_caption_length: Maximum length for each segment caption (in tokens)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()
        self.max_caption_length = max_caption_length

        # Parameters for more diverse and specific captions
        self.use_nucleus_sampling = True  # Use sampling for diversity
        self.top_p = 0.92  # Nucleus sampling threshold
        self.repetition_penalty = 1.3  # Reduce repetition
        self.min_length = 20  # Force more detailed descriptions

        print(f"Initializing AudioCaptioner with {model_name} on {self.device}")
        print(f"  Max caption length per segment: {max_caption_length} tokens")
        print(f"  Sampling: nucleus (top_p={self.top_p}), repetition_penalty={self.repetition_penalty}")

        self._init_lpmusiccaps()

    def _init_lpmusiccaps(self):
        """Initialize LP-MusicCaps model"""
        try:
            import sys
            from pathlib import Path
            from omegaconf import OmegaConf

            print("Loading LP-MusicCaps (seungheondoh/lp-music-caps) model")

            # Add lpmc to path
            lpmc_path = Path(__file__).parent.parent.parent / "lpmc"
            if lpmc_path.exists():
                sys.path.insert(0, str(lpmc_path.parent))

            # Import LP-MusicCaps components
            from lpmc.music_captioning.model.bart import BartCaptionModel
            from lpmc.utils.eval_utils import load_pretrained

            # Setup model paths
            checkpoint_path = Path.home() / ".cache" / "lp-music-caps" / "transfer.pth"
            config_path = Path.home() / ".cache" / "lp-music-caps" / "transfer.yaml"

            if not checkpoint_path.exists() or not config_path.exists():
                print(f"Downloading LP-MusicCaps checkpoint and config...")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                import urllib.request
                # Download checkpoint
                checkpoint_url = "https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth"
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

                # Download config
                config_url = "https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.yaml"
                urllib.request.urlretrieve(config_url, config_path)
                print("Checkpoint and config downloaded successfully")

            # Load config
            config = OmegaConf.load(config_path)

            # Create model
            self.model = BartCaptionModel(max_length=config.get('max_length', 128))

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device).eval()

            self.caption_type = "lpmusiccaps"
            self.num_beams = 5  # For beam search
            print("LP-MusicCaps model loaded successfully")

        except Exception as e:
            print(f"Failed to load LP-MusicCaps model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize LP-MusicCaps: {e}")

    def caption_audio(self, audio_path: Union[str, Path],
                     custom_prompt: Optional[str] = None,
                     target_duration: Optional[float] = None,
                     target_total_length: Optional[int] = None) -> str:
        """
        Generate caption for audio file using LP-MusicCaps

        Args:
            audio_path: Path to audio file
            custom_prompt: Custom prompt for generation (not used)
            target_duration: Duration to caption (in seconds). If None, captions full file
            target_total_length: Target total caption length in tokens. If provided, will summarize

        Returns:
            Generated caption string
        """
        audio_path = str(audio_path)
        return self._lpmusiccaps_caption(audio_path, target_duration, target_total_length)

    def _lpmusiccaps_caption(self, audio_path: str,
                             target_duration: Optional[float] = None,
                             target_total_length: Optional[int] = None) -> str:
        """
        Generate caption using LP-MusicCaps model

        Args:
            audio_path: Path to audio file
            target_duration: Duration to caption (in seconds)
            target_total_length: Target total caption length in tokens

        Returns:
            Caption string
        """
        print(f"Generating caption with LP-MusicCaps model for {Path(audio_path).name}")
        if target_duration:
            print(f"  Target duration: {target_duration}s")
        if target_total_length:
            print(f"  Target total length: {target_total_length} tokens")

        # Load and preprocess audio (LP-MusicCaps processes 10s chunks)
        # IMPORTANT: LP-MusicCaps model expects EXACTLY 10 seconds (160000 samples at 16kHz)
        duration = 10  # seconds per segment
        target_sr = 16000
        n_samples = int(duration * target_sr)  # = 160000 samples

        # Load audio (with optional duration limit)
        load_duration = target_duration if target_duration else None
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=load_duration)

        # Calculate number of complete 10-second chunks
        audio_length = audio.shape[-1]
        num_chunks = max(1, audio_length // n_samples)

        # If audio is shorter than 10s, pad to exactly 10s
        if audio_length < n_samples:
            audio = np.pad(audio, (0, n_samples - audio_length), mode='constant', constant_values=0)
            num_chunks = 1

        # Process only complete 10-second chunks
        # Each chunk must be EXACTLY n_samples (160000) long
        audio_chunks = []
        for i in range(num_chunks):
            start_idx = i * n_samples
            end_idx = start_idx + n_samples
            chunk = audio[start_idx:end_idx]

            # Ensure chunk is exactly the right size
            if len(chunk) == n_samples:
                audio_chunks.append(chunk)

        # If no complete chunks, create one padded chunk
        if len(audio_chunks) == 0:
            chunk = audio[:n_samples] if len(audio) >= n_samples else np.pad(audio, (0, n_samples - len(audio)), mode='constant')
            audio_chunks.append(chunk)

        # Stack into tensor - each chunk must be exactly 160000 samples
        audio_tensor = torch.from_numpy(np.stack(audio_chunks).astype('float32'))

        # Verify shape: (num_chunks, 160000)
        assert audio_tensor.shape[1] == n_samples, f"Expected {n_samples} samples per chunk, got {audio_tensor.shape[1]}"

        if self.device == 'cuda':
            audio_tensor = audio_tensor.cuda(non_blocking=True)

        # Calculate per-segment length if total target is specified
        per_segment_max_length = self.max_caption_length
        if target_total_length and len(audio_chunks) > 0:
            # Reserve space for timestamps and spacing
            effective_total = target_total_length - (len(audio_chunks) * 5)  # 5 tokens per timestamp
            per_segment_max_length = max(20, effective_total // len(audio_chunks))
            print(f"  Adjusted per-segment length: {per_segment_max_length} tokens")

        # Generate captions for each chunk with diversity parameters
        with torch.no_grad():
            outputs = self.model.generate(
                samples=audio_tensor,
                use_nucleus_sampling=self.use_nucleus_sampling,
                num_beams=self.num_beams if not self.use_nucleus_sampling else 1,
                max_length=per_segment_max_length,
                min_length=self.min_length,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

        # Combine captions from all chunks
        captions = []
        for chunk_idx, text in enumerate(outputs):
            time_stamp = f"[{chunk_idx * 10}s-{(chunk_idx + 1) * 10}s]"
            captions.append(f"{time_stamp} {text}")

        # Join all captions
        full_caption = " ".join(captions)

        print(f"Generated caption length: {len(full_caption)} characters ({len(outputs)} segments)")

        # If target_total_length is specified and we exceeded it, apply smart summarization
        if target_total_length:
            full_caption = self._smart_summarize(captions, target_total_length)
            print(f"After summarization: {len(full_caption)} characters")

        return full_caption

    def _smart_summarize(self, segment_captions: list, target_length: int) -> str:
        """
        Intelligently summarize segment captions without using an LLM
        Removes timestamps, deduplicates, and intelligently truncates

        Args:
            segment_captions: List of "[Xs-Ys] caption" strings
            target_length: Target character length (approximate)

        Returns:
            Summarized caption string
        """
        import re

        # Extract just the text (remove timestamps)
        texts = [re.sub(r'\[\d+s-\d+s\]\s*', '', cap) for cap in segment_captions]

        # Split all text into sentences
        all_sentences = []
        for text in texts:
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
            all_sentences.extend(sentences)

        # Deduplicate while preserving order (use case-insensitive comparison)
        seen = set()
        unique_sentences = []
        for sent in all_sentences:
            sent_lower = sent.lower().strip()
            if sent_lower and sent_lower not in seen:
                seen.add(sent_lower)
                unique_sentences.append(sent)

        # Join sentences and truncate to target length
        if not unique_sentences:
            return "Instrumental music"

        # Build caption by adding sentences until we reach target length
        caption_parts = []
        current_length = 0

        for sent in unique_sentences:
            # Add period if needed
            sent_with_period = sent if sent.endswith('.') else sent + '.'
            sent_length = len(sent_with_period) + 1  # +1 for space

            # If adding this sentence would exceed target, decide whether to include it
            if current_length + sent_length > target_length:
                # If we have at least one sentence, stop here
                if caption_parts:
                    break
                # Otherwise, include this sentence but truncate it
                else:
                    remaining = target_length - current_length
                    truncated = sent[:remaining].rsplit(' ', 1)[0]
                    if truncated:
                        caption_parts.append(truncated + '.')
                    break

            caption_parts.append(sent_with_period)
            current_length += sent_length

        result = ' '.join(caption_parts)

        # Ensure it ends with proper punctuation
        if result and not result.endswith('.'):
            result += '.'

        return result if result else unique_sentences[0][:target_length]

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

    def save_captions(self, captions: dict, output_file: str):
        """
        Save captions to file

        Args:
            captions: Dictionary of captions
            output_file: Output file path
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for audio_path, caption in captions.items():
                f.write(f"File: {Path(audio_path).name}\n")
                f.write(f"Caption: {caption}\n")
                f.write("-" * 80 + "\n\n")

        print(f"Captions saved to {output_file}")


if __name__ == "__main__":
    # Test captioner with LP-MusicCaps
    captioner = AudioCaptioner(model_name="lpmusiccaps")

    # Test on target files
    target_dir = Path("./fundwotsai/Deep_MIR_hw2/target_music_list_60s")

    if target_dir.exists():
        audio_files = list(target_dir.glob("*.wav"))[:1]  # Test with one file first

        captions = captioner.caption_batch([str(f) for f in audio_files])
        captioner.save_captions(captions, "./outputs/captions.txt")
