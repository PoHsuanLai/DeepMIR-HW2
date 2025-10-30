"""
Controllable Music Generation
Supports MuseControlLite, MusicGen, and other models
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
import soundfile as sf


class MusicGenerator:
    """
    Wrapper for controllable text-to-music generation models
    """

    def __init__(self, model_name: str = "musecontrollite"
    , device: Optional[str] = None):
        """
        Initialize music generator

        Args:
            model_name: Model to use ('musecontrollite', 'musicgen', 'musicgen-melody', 'coco-mulla', 'jasco')
            device: Device to use (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name.lower()

        print(f"Initializing MusicGenerator with {model_name} on {self.device}")

        if self.model_name in ["musicgen", "musicgen-melody"]:
            self._init_musicgen()
        elif self.model_name == "coco-mulla":
            self._init_cocomulla()
        elif self.model_name == "jasco":
            self._init_jasco()
        else:
            raise ValueError(f"Unknown model: {model_name}. Supported: 'musicgen', 'musicgen-melody', 'coco-mulla', 'jasco'")

    def _init_musecontrollite(self):
        """Initialize MuseControlLite model from local installation"""
        import sys
        from pathlib import Path

        # Add MuseControlLite to path
        muse_path = Path(__file__).parent.parent.parent / "MuseControlLite"
        if not muse_path.exists():
            raise RuntimeError(f"MuseControlLite not found at {muse_path}. Please clone it: git clone https://github.com/fundwotsai2001/MuseControlLite")

        sys.path.insert(0, str(muse_path))
        self.muse_path = muse_path

        try:
            from MuseControlLite_setup import setup_MuseControlLite, initialize_condition_extractors, process_musical_conditions

            print("Loading MuseControlLite from local installation")

            # Config following the woSDD-all checkpoint which supports dynamics+rhythm+melody_mono
            ckpt_base = muse_path / "checkpoints" / "woSDD-all"
            self.musecontrol_config = {
                "weight_dtype": "fp32",
                "apadapter": True,
                "condition_type": ["dynamics", "rhythm", "melody"],
                "adapter_name": "adapter",
                "GPU_id": "0",
                "transformer_ckpt_musical": str(ckpt_base / "model_3.safetensors"),
                "extractor_ckpt_musical": {
                    "dynamics": str(ckpt_base / "model_1.safetensors"),
                    "melody": str(ckpt_base / "model.safetensors"),
                    "rhythm": str(ckpt_base / "model_2.safetensors"),
                },
                "ap_scale": 1.0,
                "sigma_min": 0.3,
                "sigma_max": 500,
                # Mask parameters (set to 0 for no masking)
                "audio_mask_start_seconds": 0,
                "audio_mask_end_seconds": 0,
                "musical_attribute_mask_start_seconds": 0,
                "musical_attribute_mask_end_seconds": 0,
            }

            weight_dtype = torch.float32

            # Initialize extractors and model
            self.condition_extractors, transformer_ckpt = initialize_condition_extractors(self.musecontrol_config)
            self.model = setup_MuseControlLite(self.musecontrol_config, weight_dtype, transformer_ckpt)
            self.model = self.model.to(self.device)

            # Store the processing function
            self.process_musical_conditions = process_musical_conditions

            self.sample_rate = 44100
            self.max_duration = 47.0

            print(f"MuseControlLite loaded successfully (sample_rate={self.sample_rate})")

        except Exception as e:
            print(f"Error loading MuseControlLite: {e}")
            raise RuntimeError(f"Failed to load MuseControlLite model: {e}") from e

    def _init_musicgen(self):
        """Initialize MusicGen model via Hugging Face transformers"""
        try:
            from transformers import MusicgenForConditionalGeneration, MusicgenMelodyForConditionalGeneration, AutoProcessor

            if self.model_name == "musicgen-melody":
                model_id = "facebook/musicgen-melody"
                self.supports_melody = True
                # Use the MusicgenMelody specific class
                print(f"Loading MusicGen-Melody from Hugging Face: {model_id}")
                self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id)
                self.processor = AutoProcessor.from_pretrained(model_id)
            else:
                # Use small model for faster loading, can change to medium/large
                model_id = "facebook/musicgen-small"
                self.supports_melody = False
                print(f"Loading MusicGen from Hugging Face: {model_id}")
                self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)
                self.processor = AutoProcessor.from_pretrained(model_id)

            # Move to device
            self.model = self.model.to(self.device)

            self.sample_rate = self.model.config.audio_encoder.sampling_rate
            self.max_duration = 30.0  # MusicGen default

            # Initialize source separator for advanced conditioning
            self.source_separator = None
            self.use_source_separation = True  # Enable by default

            if self.use_source_separation:
                try:
                    from src.extraction.source_separator import SourceSeparator
                    self.source_separator = SourceSeparator(device=self.device)
                    print("Source separation enabled (Demucs)")
                except Exception as e:
                    print(f"Warning: Could not load source separator: {e}")
                    self.source_separator = None

            print(f"MusicGen loaded successfully (sample_rate={self.sample_rate})")

        except Exception as e:
            print(f"Error loading MusicGen: {e}")
            raise RuntimeError(f"Failed to load MusicGen model: {e}") from e

    def _init_cocomulla(self):
        """Initialize Coco-Mulla model from local repository"""
        import sys
        from pathlib import Path

        # Add coco-mulla-repo to path
        cocomulla_path = Path(__file__).parent.parent.parent / "coco-mulla-repo"
        if not cocomulla_path.exists():
            raise RuntimeError(f"coco-mulla-repo not found at {cocomulla_path}")

        sys.path.insert(0, str(cocomulla_path))
        self.cocomulla_path = cocomulla_path

        try:
            from coco_mulla.models import CoCoMulla
            from coco_mulla.utilities import get_device
            from config import TrainCfg

            print("Loading Coco-Mulla from local repository")

            # Model configuration - matching README inference example
            self.cocomulla_config = {
                'sample_sec': 20,  # Generate 20 second clips
                'num_layers': 48,  # From README example
                'latent_dim': 12,  # From README example
                'sample_rate': 32000,
                'frame_res': 50,
            }

            # Initialize model (parameters: sec, num_layers, latent_dim)
            self.model = CoCoMulla(
                sec=self.cocomulla_config['sample_sec'],
                num_layers=self.cocomulla_config['num_layers'],
                latent_dim=self.cocomulla_config['latent_dim']
            ).to(self.device)

            # Load pretrained weights (r=0.2)
            checkpoint_path = cocomulla_path / "checkpoints" / "model_weights" / "r_12_sampling_prob-based_0.0_0.2_lr_2e-3_dataset_2" / "diff_9_end.pth"
            if checkpoint_path.exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                import torch
                checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
                print("Checkpoint loaded successfully")
            else:
                print("Warning: No checkpoint found, using initialized weights")

            self.model.eval()

            self.sample_rate = self.cocomulla_config['sample_rate']
            self.max_duration = self.cocomulla_config['sample_sec']

            # Initialize source separator for drum extraction
            self.source_separator = None
            try:
                from src.extraction.source_separator import SourceSeparator
                self.source_separator = SourceSeparator(device=self.device)
                print("Source separation enabled (Demucs)")
            except Exception as e:
                print(f"Warning: Could not load source separator: {e}")

            # Initialize chord extractor
            self.chord_extractor = None
            try:
                from src.extraction.chord_extractor import ChordExtractor
                self.chord_extractor = ChordExtractor()
                print("Chord extraction enabled")
            except Exception as e:
                print(f"Warning: Could not load chord extractor: {e}")

            # Initialize melody extractor for MIDI extraction
            self.melody_extractor = None
            try:
                from src.extraction.melody_extractor import MelodyExtractor
                self.melody_extractor = MelodyExtractor()
                print("Melody extraction enabled")
            except Exception as e:
                print(f"Warning: Could not load melody extractor: {e}")

            print(f"Coco-Mulla loaded successfully (sample_rate={self.sample_rate})")

        except Exception as e:
            print(f"Error loading Coco-Mulla: {e}")
            raise RuntimeError(f"Failed to load Coco-Mulla model: {e}") from e

    def generate(self,
                text_prompt: str,
                melody: Optional[np.ndarray] = None,
                rhythm: Optional[np.ndarray] = None,
                dynamics: Optional[np.ndarray] = None,
                audio_condition: Optional[str] = None,
                duration: float = 30.0,
                cfg_scale: float = 3.0,
                temperature: float = 1.0,
                **kwargs) -> np.ndarray:
        """
        Generate music with various conditions

        Args:
            text_prompt: Text description of desired music
            melody: Melody condition (optional) - for MuseControlLite, ignored (uses audio_condition)
            rhythm: Rhythm condition (optional) - for MuseControlLite, ignored (uses audio_condition)
            dynamics: Dynamics condition (optional) - for MuseControlLite, ignored (uses audio_condition)
            audio_condition: Audio file path for conditioning
            duration: Generation duration in seconds
            cfg_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            Generated audio as numpy array
        """
        if self.model_name == "jasco":
            return self._generate_jasco(
                text_prompt, audio_condition, duration,
                cfg_scale, temperature, **kwargs
            )
        elif self.model_name == "coco-mulla":
            return self._generate_cocomulla(
                text_prompt, audio_condition, duration,
                cfg_scale, temperature, **kwargs
            )
        elif self.model_name in ["musicgen", "musicgen-melody"]:
            return self._generate_musicgen(
                text_prompt, audio_condition, duration,
                cfg_scale, temperature, **kwargs
            )
        elif self.model_name == "musecontrollite":
            return self._generate_musecontrollite(
                text_prompt, audio_condition,
                duration, cfg_scale, temperature, **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _generate_musecontrollite(self,
                                  text_prompt: str,
                                  audio_file: Optional[str],
                                  duration: float,
                                  cfg_scale: float,
                                  temperature: float,
                                  **kwargs) -> np.ndarray:
        """
        Generate using MuseControlLite

        Args:
            text_prompt: Text description
            audio_file: Path to audio file for extracting conditions
            duration: Duration in seconds
            cfg_scale: Guidance scale
            temperature: Temperature (unused for MuseControlLite)

        Returns:
            Generated audio as numpy array
        """
        import tempfile

        print(f"Generating with MuseControlLite:")
        print(f"  Text: {text_prompt}")
        print(f"  Audio condition: {audio_file if audio_file else 'None'}")
        print(f"  CFG Scale: {cfg_scale}")

        if audio_file is None:
            raise ValueError("MuseControlLite requires audio_condition (source audio file path)")

        # Create temporary output directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process musical conditions using the official MuseControlLite function
            # This extracts melody, rhythm, and dynamics from the audio file
            final_condition, final_condition_audio = self.process_musical_conditions(
                self.musecontrol_config,
                audio_file,
                self.condition_extractors,
                temp_dir,
                0,  # index
                torch.float32,
                self.model
            )

            # Generate with MuseControlLite
            with torch.no_grad():
                waveform = self.model(
                    extracted_condition=final_condition,
                    extracted_condition_audio=final_condition_audio,
                    prompt=text_prompt,
                    negative_prompt="low quality, distorted",
                    num_inference_steps=100,
                    guidance_scale_text=cfg_scale,
                    guidance_scale_con=cfg_scale,
                    guidance_scale_audio=0.0,
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=min(duration, self.max_duration),
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).audios

        # Convert to numpy
        audio_np = waveform[0].T.float().cpu().numpy()
        # Convert stereo to mono if needed
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)

        print(f"  Generated {len(audio_np) / self.sample_rate:.2f}s of audio")

        return audio_np

    def _generate_musicgen(self,
                          text_prompt: str,
                          audio_condition: Optional[str],
                          duration: float,
                          cfg_scale: float,
                          temperature: float,
                          **kwargs) -> np.ndarray:
        """
        Generate using MusicGen

        Args:
            Similar to generate()

        Returns:
            Generated audio
        """
        print(f"Generating with MusicGen (Hugging Face):")
        print(f"  Text: {text_prompt[:200]}...")  # Show first 200 chars
        print(f"  Duration: {duration}s")
        print(f"  CFG Scale: {cfg_scale}")
        print(f"  Melody conditioning: {'Yes' if (self.supports_melody and audio_condition) else 'No'}")
        print(f"  Source separation: {'Yes' if self.source_separator else 'No'}")

        # IMPORTANT: Truncate text to avoid CUDA errors
        # MusicGen-Melody has a context limit, and melody conditioning takes up space
        # When using melody conditioning, text must be much shorter to avoid exceeding max_new_tokens
        tokenizer = self.processor.tokenizer

        # Use shorter max_length when melody conditioning is active
        # Melody conditioning uses a prefix that takes up sequence space
        if self.supports_melody and audio_condition:
            max_text_length = 256  # Reduced from 512 to leave room for melody prefix
        else:
            max_text_length = 512

        encoded = tokenizer(text_prompt, truncation=True, max_length=max_text_length)
        text_prompt = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
        print(f"  Truncated text length: {len(tokenizer(text_prompt)['input_ids'])} tokens")

        try:
            # Apply source separation if enabled
            audio_for_conditioning = audio_condition
            temp_file = None

            if self.supports_melody and audio_condition and self.source_separator:
                print("  Applying source separation (Demucs)...")
                try:
                    # Get instrumental mix (no vocals) for better melody conditioning
                    instrumental = self.source_separator.get_melodic_instruments(audio_condition)

                    # Save to temp file
                    temp_file = self.source_separator.save_stem_as_tempfile(
                        instrumental,
                        sr=self.sample_rate
                    )
                    audio_for_conditioning = temp_file
                    print(f"  Using separated melodic instruments")
                except Exception as e:
                    print(f"  Warning: Source separation failed, using original audio: {e}")
                    audio_for_conditioning = audio_condition

            # Process inputs using the processor
            # For MusicGen-melody, use audio parameter in processor
            if self.supports_melody and audio_for_conditioning:
                import librosa
                import soundfile as sf

                # CRITICAL FIX: Load audio for conditioning with SAME duration as generation
                # This ensures melody conditioning is aligned with generated output
                melody_audio, sr = librosa.load(
                    audio_for_conditioning,
                    sr=self.sample_rate,
                    mono=True,
                    duration=duration  # Match generation duration!
                )

                # Process with both text and audio
                inputs = self.processor(
                    audio=melody_audio,
                    sampling_rate=self.sample_rate,
                    text=[text_prompt],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            else:
                # Text-only
                inputs = self.processor(
                    text=[text_prompt],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Calculate max tokens based on duration
            # MusicGen generates at 50 Hz (50 tokens per second)
            max_new_tokens = int(duration * 50)

            # Generate
            print(f"  Generating with {max_new_tokens} tokens...")
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=cfg_scale,
                    temperature=temperature,
                )

            # Clean up temp file
            if temp_file:
                import os
                try:
                    os.unlink(temp_file)
                except:
                    pass

            # Convert to numpy
            audio_np = audio_values[0, 0].cpu().numpy()

            print(f"  Generated {len(audio_np) / self.sample_rate:.2f}s of audio")

        except Exception as e:
            print(f"Error during generation: {e}")
            raise RuntimeError(f"Failed to generate music: {e}") from e

        return audio_np

    def _generate_cocomulla(self,
                           text_prompt: str,
                           audio_condition: Optional[str],
                           duration: float,
                           cfg_scale: float,
                           temperature: float,
                           **kwargs) -> np.ndarray:
        """
        Generate using Coco-Mulla with chord + drum + MIDI conditioning

        Args:
            text_prompt: Text description
            audio_condition: Path to audio file for extracting chords/drums/MIDI
            duration: Duration (Coco-Mulla generates 20s clips)
            cfg_scale: Guidance scale
            temperature: Temperature (unused for Coco-Mulla)

        Returns:
            Generated audio as numpy array
        """
        print("Generating with Coco-Mulla:")
        print(f"  Text: {text_prompt}")
        print(f"  Audio condition: {audio_condition if audio_condition else 'None'}")
        print(f"  CFG Scale: {cfg_scale}")
        print(f"  Duration: {min(duration, 20.0)}s (Coco-Mulla max)")

        if audio_condition is None:
            raise ValueError("Coco-Mulla requires audio_condition for extracting chords, drums, and MIDI")

        try:
            import sys
            sys.path.insert(0, str(self.cocomulla_path))

            from coco_mulla.utilities.encodec_utils import extract_rvq
            from coco_mulla.utilities import np2torch
            import librosa

            sr = self.cocomulla_config['sample_rate']
            res = self.cocomulla_config['frame_res']
            sample_sec = self.max_duration  # MUST use exactly 20s (model's trained duration)

            print(f"  Using Coco-Mulla max duration: {sample_sec}s")

            # Load FULL audio (don't truncate yet - let extract_rvq handle it)
            wav, _ = librosa.load(audio_condition, sr=sr, mono=True, duration=sample_sec)
            wav = np2torch(wav).to(self.device)[None, None, ...]

            print("  Separating drums and extracting RVQ...")
            # Use Coco-Mulla's built-in separation and extraction
            from coco_mulla.utilities.sep_utils import separate
            wavs = separate(wav, sr)
            drums_rvq = extract_rvq(wavs["drums"], sr=sr)

            # drums_rvq comes out as [4, time], add batch dim
            if drums_rvq.dim() == 2:
                drums_rvq = drums_rvq.unsqueeze(0)  # [1, 4, time]

            print(f"  drums_rvq shape: {drums_rvq.shape}")

            # CRITICAL: The checkpoint was trained with max_n_frames + 1 = 1001 frames
            # Model code says max_n_frames = 1000, but embeddings are (1001, ...)
            # So we need 1001 frames, not 1000!
            max_n_frames = int(sample_sec * res) + 1  # 1001 for 20s (NOT 1000!)

            chord = np.zeros((1, max_n_frames, 13))
            pad_chord = np.ones((1, max_n_frames, 1))  # All padding (no chords)
            chord = np.concatenate([chord, pad_chord], -1)  # [1, 1000, 14]
            chord = torch.from_numpy(chord).to(self.device).float()

            midi = np.zeros((1, max_n_frames, 128))  # [1, 1000, 128]
            midi = torch.from_numpy(midi).to(self.device).float()

            # Crop drums_rvq to exactly max_n_frames
            if drums_rvq.shape[-1] > max_n_frames:
                drums_rvq = drums_rvq[:, :, :max_n_frames]
            elif drums_rvq.shape[-1] < max_n_frames:
                import torch.nn.functional as F
                drums_rvq = F.pad(drums_rvq, (0, max_n_frames - drums_rvq.shape[-1]), "constant", 0)

            drums_rvq = drums_rvq.to(self.device).long()

            print(f"  Final shapes - drums: {drums_rvq.shape}, chord: {chord.shape}, midi: {midi.shape}")

            # Use model's generate method (simpler and more reliable)
            print("  Generating with Coco-Mulla...")
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    piano_roll=midi,
                    desc=[text_prompt],
                    chords=chord,
                    drums=drums_rvq,
                    num_samples=1
                )

            # Decode tokens to audio using encodec
            from coco_mulla.utilities.encodec_utils import save_rvq
            import tempfile
            import os

            # Save to temp file and load back
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name

            save_rvq(output_list=[temp_path], tokens=gen_tokens)

            # Load generated audio
            audio_np, _ = librosa.load(temp_path, sr=self.sample_rate, mono=True)

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            print(f"  Generated {len(audio_np) / self.sample_rate:.2f}s of audio")

        except Exception as e:
            print(f"Error during Coco-Mulla generation: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate music with Coco-Mulla: {e}") from e

        return audio_np

    def _init_jasco(self):
        """Initialize JASCO model from HuggingFace"""
        try:
            # Save original sys.argv to prevent JASCO from parsing our arguments
            import sys
            original_argv = sys.argv.copy()
            sys.argv = [sys.argv[0]]  # Keep only script name

            from audiocraft.models import JASCO

            # Restore original arguments
            sys.argv = original_argv

            # Use 1B model
            model_id = 'facebook/jasco-chords-drums-melody-1B'
            print(f"Loading JASCO from HuggingFace: {model_id}")

            # Create chord mapping path
            assets_dir = Path(__file__).parent.parent.parent / "assets"
            assets_dir.mkdir(exist_ok=True)
            chord_mapping_path = str(assets_dir / "chord_to_index_mapping.pkl")

            # Download chord mapping if it doesn't exist
            if not Path(chord_mapping_path).exists():
                print(f"  Downloading chord mapping file...")
                from huggingface_hub import hf_hub_download
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_id,
                        filename='chord_to_index_mapping.pkl',
                        local_dir=str(assets_dir)
                    )
                    print(f"  Downloaded to: {downloaded_path}")
                except Exception as e:
                    print(f"  Could not download chord mapping: {e}")
                    # Create a basic fallback mapping
                    import pickle
                    chord_mapping = {
                        'N': 0, 'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5, 'F': 6,
                        'F#': 7, 'G': 8, 'G#': 9, 'A': 10, 'A#': 11, 'B': 12,
                    }
                    with open(chord_mapping_path, 'wb') as f:
                        pickle.dump(chord_mapping, f)
                    print(f"  Created fallback chord mapping")

            # Load JASCO model (already placed on device by get_pretrained)
            self.model = JASCO.get_pretrained(
                model_id,
                device=self.device,
                chords_mapping_path=chord_mapping_path
            )

            # Set generation parameters
            self.model.set_generation_params(
                duration=10.0,  # JASCO generates 10s clips
                cfg_coef_all=3.0,  # CFG for all conditions (best: equal weighting)
                cfg_coef_txt=3.0,  # CFG for text only (best: equal weighting)
            )

            self.sample_rate = self.model.sample_rate
            self.max_duration = 10.0  # JASCO default

            # Initialize source separator for drum/melody extraction
            self.source_separator = None
            try:
                from src.extraction.source_separator import SourceSeparator
                self.source_separator = SourceSeparator(device=self.device)
                print("Source separation enabled (Demucs)")
            except Exception as e:
                print(f"Warning: Could not load source separator: {e}")

            # Initialize chord extractor
            self.chord_extractor = None
            try:
                from src.extraction.chord_extractor import ChordExtractor
                # Disable madmom due to compatibility issues, use simple chromagram method
                self.chord_extractor = ChordExtractor(use_madmom=False)
                print("Chord extraction enabled (using chromagram method)")
            except Exception as e:
                print(f"Warning: Could not load chord extractor: {e}")

            # Initialize JASCO melody extractor
            self.melody_extractor = None
            try:
                from src.extraction.jasco_melody_extractor import JASCOMelodyExtractor
                self.melody_extractor = JASCOMelodyExtractor()
                print("JASCO melody extraction enabled")
            except Exception as e:
                print(f"Warning: Could not load JASCO melody extractor: {e}")

            print(f"JASCO loaded successfully (sample_rate={self.sample_rate})")

        except Exception as e:
            print(f"Error loading JASCO: {e}")
            raise RuntimeError(f"Failed to load JASCO model: {e}") from e

    def _generate_jasco(self,
                       text_prompt: str,
                       audio_condition: Optional[str],
                       duration: float,
                       cfg_scale: float,
                       temperature: float,
                       **kwargs) -> np.ndarray:
        """
        Generate using JASCO with chords + drums + melody conditioning

        Args:
            text_prompt: Text description
            audio_condition: Path to audio file for extracting chords/drums/melody
            duration: Duration (JASCO generates 10s clips)
            cfg_scale: Guidance scale
            temperature: Temperature (unused for JASCO)

        Returns:
            Generated audio as numpy array
        """
        print("Generating with JASCO:")
        print(f"  Text: {text_prompt}")
        print(f"  Audio condition: {audio_condition if audio_condition else 'None'}")
        print(f"  CFG Scale: {cfg_scale}")
        print(f"  Duration: {min(duration, 10.0)}s (JASCO max)")

        if audio_condition is None:
            print("  Warning: JASCO works best with audio conditioning")
            audio_condition = None

        try:
            # Extract conditions from audio
            chords = None
            drums = None
            melody = None

            if audio_condition and self.source_separator:
                print("  Applying Demucs source separation...")

                # Separate stems
                stems = self.source_separator.separate(audio_condition)

                # Extract chords from FULL MIX (better for madmom)
                if self.chord_extractor:
                    print("  Extracting chords from full mix using madmom...")
                    # Load original full mix audio
                    import librosa
                    full_mix, sr = librosa.load(audio_condition, sr=self.source_separator.model.samplerate, mono=True)

                    chords = self.chord_extractor.extract_chords_from_audio(
                        full_mix,
                        sr=sr
                    )
                    chords = self.chord_extractor.format_for_jasco(chords, duration=10.0)
                    print(f"  Extracted {len(chords)} chord changes")
                    if chords:
                        chord_names = [c[0] for c in chords[:5]]
                        print(f"    First chords: {chord_names}...")

                # Use drums stem directly
                drums_audio = stems['drums']
                # Convert to torch tensor for JASCO
                drums = torch.from_numpy(drums_audio).unsqueeze(0).to(self.device)
                print(f"  Using drums stem: {drums.shape}")

                # Extract melody salience matrix using proper F0-to-MIDI mapping
                if hasattr(self, 'melody_extractor') and self.melody_extractor:
                    print("  Extracting melody salience matrix (JASCO format)...")
                    try:
                        # Use melodic stems (bass + other) for melody extraction
                        melody_audio = stems['bass'] + stems['other']

                        # Ensure mono
                        if len(melody_audio.shape) > 1:
                            melody_audio = melody_audio.mean(axis=0)

                        # Extract melody using proper MIDI pitch mapping
                        melody = self.melody_extractor.extract_from_numpy(
                            melody_audio,
                            sample_rate=self.source_separator.model.samplerate,
                            duration=duration
                        )

                        # Move to device
                        melody = melody.to(self.device)

                        # Validate the matrix
                        self.melody_extractor.validate_matrix(melody)

                        print(f"  Using melody salience matrix: {melody.shape}")
                        print(f"    Non-zero elements: {(melody > 0).sum().item()} / {melody.numel()}")
                        print(f"    Sparsity: {100 * (melody == 0).sum().item() / melody.numel():.1f}%")
                    except Exception as e:
                        print(f"  Warning: Melody extraction failed: {e}")
                        melody = None
                else:
                    melody = None
                    print("  Skipping melody extraction (extractor not available)")

            # Update CFG parameters (JASCO duration is fixed at model initialization)
            # Prioritize melody/chords/drums over text for better Gen↔Target similarity
            self.model.set_generation_params(
                cfg_coef_all=3.0,  # Equal weighting (best configuration from experiments)
                cfg_coef_txt=3.0,   # Equal weighting (best configuration from experiments)
            )

            # Generate with JASCO
            print("  Generating...")
            with torch.no_grad():
                outputs = self.model.generate_music(
                    descriptions=[text_prompt],
                    chords=chords if chords else None,
                    drums_wav=drums if drums is not None else None,
                    melody_salience_matrix=melody if melody is not None else None,
                    progress=True,
                )

            # Convert to numpy
            audio_np = outputs[0, 0].cpu().numpy()

            print(f"  Generated {len(audio_np) / self.sample_rate:.2f}s of audio")

        except Exception as e:
            print(f"Error during JASCO generation: {e}")
            raise RuntimeError(f"Failed to generate music with JASCO: {e}") from e

        return audio_np

    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: Optional[int] = None):
        """
        Save generated audio to file

        Args:
            audio: Audio array
            output_path: Output file path
            sample_rate: Sample rate (uses model's if not specified)
        """
        sr = sample_rate or self.sample_rate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, audio, sr)
        print(f"Audio saved to {output_path}")

    def generate_and_save(self,
                         text_prompt: str,
                         output_path: str,
                         **kwargs) -> str:
        """
        Generate music and save to file

        Args:
            text_prompt: Text description
            output_path: Output file path
            **kwargs: Additional generation parameters

        Returns:
            Path to saved audio file
        """
        audio = self.generate(text_prompt, **kwargs)
        self.save_audio(audio, output_path)
        return output_path


if __name__ == "__main__":
    # Test generator
    generator = MusicGenerator(model_name="musicgen")

    # Simple generation
    text_prompt = "A gentle piano melody in C major, slow tempo, classical style"

    audio = generator.generate(
        text_prompt=text_prompt,
        duration=10.0,
        cfg_scale=7.0
    )

    print(f"Generated audio shape: {audio.shape}")

    # Save
    generator.save_audio(audio, "./outputs/generated/test_generation.wav")
