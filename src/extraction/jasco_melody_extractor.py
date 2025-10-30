"""
JASCO Melody Salience Matrix Extractor

Creates melody salience matrices in the format expected by JASCO:
- Shape: [53, n_frames] where n_frames = duration_sec * 50 (50Hz frame rate)
- 53 bins represent MIDI notes 36-88 (C2 to E6)
- Values are float32 in range [0, 1] representing pitch salience/probability
"""

import torch
import numpy as np
import librosa
import tempfile
import soundfile as sf
from typing import Optional
from pathlib import Path


class JASCOMelodyExtractor:
    """Extract melody salience matrices for JASCO conditioning using basic_pitch"""

    def __init__(self, sample_rate: int = 22050, method: str = "basic_pitch"):
        """
        Initialize melody extractor.

        Args:
            sample_rate: Sample rate for audio loading
            method: Extraction method ("basic_pitch" or "f0")
        """
        self.sample_rate = sample_rate
        self.midi_start = 36  # C2
        self.midi_end = 88    # E6
        self.n_bins = 53      # Number of pitch bins
        self.frame_rate = 50  # Hz (JASCO's frame rate)
        self.method = method

        # Load basic_pitch model if using that method
        if self.method == "basic_pitch":
            try:
                from basic_pitch import ICASSP_2022_MODEL_PATH
                from basic_pitch.inference import predict
                self.basic_pitch_model = ICASSP_2022_MODEL_PATH
                self.basic_pitch_predict = predict
                print("basic_pitch loaded successfully")
            except ImportError:
                print("Warning: basic_pitch not available, falling back to F0 method")
                self.method = "f0"

    def extract_from_audio(
        self,
        audio_path: str,
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Extract melody salience matrix from audio file.

        Args:
            audio_path: Path to audio file
            duration: Duration to process (seconds)

        Returns:
            torch.Tensor: Melody salience matrix [53, n_frames]
        """
        if self.method == "basic_pitch":
            return self._extract_basic_pitch(audio_path, duration)
        else:
            return self._extract_f0(audio_path, duration)

    def _extract_basic_pitch(
        self,
        audio_path: str,
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Extract melody using basic_pitch neural network.

        Args:
            audio_path: Path to audio file
            duration: Duration to process (seconds)

        Returns:
            torch.Tensor: Melody salience matrix [53, n_frames]
        """
        # Run basic_pitch prediction
        model_output, midi_data, note_events = self.basic_pitch_predict(
            audio_path,
            self.basic_pitch_model,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,  # milliseconds
            minimum_frequency=librosa.midi_to_hz(self.midi_start),  # C2 (65.41 Hz)
            maximum_frequency=librosa.midi_to_hz(self.midi_end),     # E6 (1318.51 Hz)
            melodia_trick=True,  # Better monophonic melody extraction
        )

        # Extract note posteriorgram (pitch probabilities over time)
        # Shape: [n_frames_original, 88] for MIDI notes 21-108
        note_posteriorgram = model_output['note']

        # Extract MIDI range 36-88 (C2 to E6)
        # basic_pitch outputs MIDI 21-108, so we need indices 15-67
        # (36-21=15, 88-21=67, end index is exclusive so 68)
        start_idx = self.midi_start - 21
        end_idx = self.midi_end - 21 + 1
        melody_salience = note_posteriorgram[:, start_idx:end_idx]  # [n_frames_original, 53]

        # Transpose to [53, n_frames]
        melody_salience = melody_salience.T

        # Resample to 50Hz (JASCO's frame rate)
        n_frames_target = int(duration * self.frame_rate)
        n_frames_original = melody_salience.shape[1]

        if n_frames_original != n_frames_target:
            melody_salience_resampled = np.zeros((self.n_bins, n_frames_target), dtype=np.float32)
            for i in range(self.n_bins):
                melody_salience_resampled[i] = np.interp(
                    np.linspace(0, n_frames_original - 1, n_frames_target),
                    np.arange(n_frames_original),
                    melody_salience[i]
                )
            melody_salience = melody_salience_resampled

        # Convert to torch tensor
        melody_tensor = torch.from_numpy(melody_salience).float()

        # Ensure values are in [0, 1] range
        melody_tensor = torch.clamp(melody_tensor, 0, 1)

        # Apply salience threshold to keep only strong pitches
        # This makes the matrix sparse, similar to F0 extraction
        # Threshold based on JASCO's expectation of salient melody notes
        salience_threshold = 0.3  # Keep only confident pitch activations
        melody_tensor[melody_tensor < salience_threshold] = 0.0

        return melody_tensor

    def _extract_f0(
        self,
        audio_path: str,
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Extract melody using F0 extraction (fallback method).

        Args:
            audio_path: Path to audio file
            duration: Duration to process (seconds)

        Returns:
            torch.Tensor: Melody salience matrix [53, n_frames]
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)

        # Extract F0 using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.midi_to_hz(self.midi_start),   # C2 (65.41 Hz)
            fmax=librosa.midi_to_hz(self.midi_end),     # E6 (1318.51 Hz)
            sr=sr,
            hop_length=int(sr / self.frame_rate)  # To get 50Hz frame rate
        )

        # Convert F0 to MIDI note numbers
        midi_notes = librosa.hz_to_midi(f0)

        # Create salience matrix
        n_frames = len(f0)
        melody_salience = np.zeros((self.n_bins, n_frames), dtype=np.float32)

        # Map MIDI notes to bins
        for t in range(n_frames):
            if voiced_flag[t] and not np.isnan(midi_notes[t]):
                # Find closest MIDI note
                midi_note = int(np.round(midi_notes[t]))
                if self.midi_start <= midi_note <= self.midi_end:
                    bin_idx = midi_note - self.midi_start
                    # Use voiced probability as salience
                    melody_salience[bin_idx, t] = voiced_probs[t]

        # Resample to exactly match JASCO's expected frame count
        n_frames_target = int(duration * self.frame_rate)
        if n_frames != n_frames_target:
            melody_salience_resampled = np.zeros(
                (self.n_bins, n_frames_target),
                dtype=np.float32
            )
            for i in range(self.n_bins):
                melody_salience_resampled[i] = np.interp(
                    np.linspace(0, n_frames - 1, n_frames_target),
                    np.arange(n_frames),
                    melody_salience[i]
                )
            melody_salience = melody_salience_resampled

        # Convert to torch tensor
        melody_tensor = torch.from_numpy(melody_salience).float()

        # Ensure values are in [0, 1] range
        melody_tensor = torch.clamp(melody_tensor, 0, 1)

        return melody_tensor

    def extract_from_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Extract melody salience matrix from numpy audio array.

        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate of the audio
            duration: Duration to process (seconds)

        Returns:
            torch.Tensor: Melody salience matrix [53, n_frames]
        """
        # For basic_pitch, we need to save to a temp file
        if self.method == "basic_pitch":
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, sample_rate)

            try:
                # Extract using basic_pitch
                melody = self._extract_basic_pitch(tmp_path, duration)
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

            return melody
        else:
            # Use F0 extraction method
            return self._extract_f0_from_numpy(audio, sample_rate, duration)

    def _extract_f0_from_numpy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration: float = 10.0
    ) -> torch.Tensor:
        """
        Extract melody using F0 from numpy array (fallback method).

        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate of the audio
            duration: Duration to process (seconds)

        Returns:
            torch.Tensor: Melody salience matrix [53, n_frames]
        """
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )
            sample_rate = self.sample_rate

        # Trim to duration
        max_samples = int(duration * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract F0 using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.midi_to_hz(self.midi_start),
            fmax=librosa.midi_to_hz(self.midi_end),
            sr=sample_rate,
            hop_length=int(sample_rate / self.frame_rate)
        )

        # Convert F0 to MIDI note numbers
        midi_notes = librosa.hz_to_midi(f0)

        # Create salience matrix
        n_frames = len(f0)
        melody_salience = np.zeros((self.n_bins, n_frames), dtype=np.float32)

        # Map MIDI notes to bins
        for t in range(n_frames):
            if voiced_flag[t] and not np.isnan(midi_notes[t]):
                midi_note = int(np.round(midi_notes[t]))
                if self.midi_start <= midi_note <= self.midi_end:
                    bin_idx = midi_note - self.midi_start
                    melody_salience[bin_idx, t] = voiced_probs[t]

        # Resample to exactly match JASCO's expected frame count
        n_frames_target = int(duration * self.frame_rate)
        if n_frames != n_frames_target:
            melody_salience_resampled = np.zeros(
                (self.n_bins, n_frames_target),
                dtype=np.float32
            )
            for i in range(self.n_bins):
                melody_salience_resampled[i] = np.interp(
                    np.linspace(0, n_frames - 1, n_frames_target),
                    np.arange(n_frames),
                    melody_salience[i]
                )
            melody_salience = melody_salience_resampled

        # Convert to torch tensor
        melody_tensor = torch.from_numpy(melody_salience).float()
        melody_tensor = torch.clamp(melody_tensor, 0, 1)

        return melody_tensor

    def validate_matrix(self, melody: torch.Tensor) -> bool:
        """
        Validate that a melody salience matrix has the correct format.

        Args:
            melody: Melody salience matrix

        Returns:
            bool: True if valid

        Raises:
            ValueError: If matrix is invalid
        """
        if len(melody.shape) != 2:
            raise ValueError(f"Matrix must be 2D, got shape {melody.shape}")

        if melody.shape[0] != self.n_bins:
            raise ValueError(
                f"First dimension must be {self.n_bins}, got {melody.shape[0]}"
            )

        if melody.dtype != torch.float32:
            raise ValueError(f"Dtype must be float32, got {melody.dtype}")

        if melody.min() < 0 or melody.max() > 1:
            raise ValueError(
                f"Values must be in [0, 1], got range [{melody.min():.3f}, {melody.max():.3f}]"
            )

        duration = melody.shape[1] / self.frame_rate
        print(f"✓ Valid melody matrix for {duration:.1f}s audio")
        return True


if __name__ == "__main__":
    # Test the extractor
    import sys

    if len(sys.argv) < 2:
        print("Usage: python jasco_melody_extractor.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(f"Extracting melody from: {audio_path}")
    print(f"Duration: {duration}s")

    extractor = JASCOMelodyExtractor()
    melody = extractor.extract_from_audio(audio_path, duration)

    print(f"\nMelody salience matrix:")
    print(f"  Shape: {melody.shape}")
    print(f"  Dtype: {melody.dtype}")
    print(f"  Value range: [{melody.min():.3f}, {melody.max():.3f}]")
    print(f"  Non-zero elements: {(melody > 0).sum().item()} / {melody.numel()}")
    print(f"  Sparsity: {100 * (melody == 0).sum().item() / melody.numel():.1f}%")

    # Validate
    extractor.validate_matrix(melody)

    # Save
    output_path = "test_melody_salience.pt"
    torch.save(melody, output_path)
    print(f"\nSaved to: {output_path}")
