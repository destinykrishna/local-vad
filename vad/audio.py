"""Audio loading, validation, and framing module."""

from dataclasses import dataclass
from typing import Iterator, Tuple
import wave
import numpy as np

from vad.config import VADConfig


@dataclass
class AudioFrame:
    """Represents a discrete audio frame for feature analysis.

    Attributes:
        data: 1D numpy array of float32 samples in range [-1.0, 1.0].
        timestamp_start_s: Start time in seconds relative to the audio origin.
        timestamp_end_s: End time in seconds relative to the audio origin.
        frame_index: Sequential zero-based index of the frame.
        sample_rate: Sample rate in Hz.
    """

    data: np.ndarray
    timestamp_start_s: float
    timestamp_end_s: float
    frame_index: int
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        """Frame duration in seconds."""
        return self.timestamp_end_s - self.timestamp_start_s

    @property
    def num_samples(self) -> int:
        """Number of samples in this frame."""
        return len(self.data)


def load_wav(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a mono 16 kHz PCM WAV file and convert to normalized float32.

    Args:
        file_path: Path to the .wav audio file.
        target_sr: Expected sample rate (default: 16000 Hz).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is 1D float32 in [-1.0, 1.0].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If audio format is invalid (corrupted, non-WAV, multi-channel,
                    unexpected sample rate, or unsupported bit depth).
    """
    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        with wave.open(file_path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()

            if num_channels != 1:
                raise ValueError(f"Expected mono WAV file, got {num_channels} channels.")

            if sample_rate != target_sr:
                raise ValueError(f"Expected sample rate {target_sr} Hz, got {sample_rate} Hz.")

            if num_frames == 0:
                return np.empty(0, dtype=np.float32), sample_rate

            raw_bytes = wf.readframes(num_frames)

            if sample_width == 2:
                # 16-bit signed PCM
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                # 8-bit unsigned PCM
                audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 4:
                # 32-bit signed PCM
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(
                    f"Unsupported sample width: {sample_width} bytes ({sample_width * 8}-bit). "
                    f"Expected 16-bit PCM."
                )
    except wave.Error as exc:
        raise ValueError(f"Invalid or corrupt WAV file '{file_path}': {exc}") from exc
    except EOFError as exc:
        raise ValueError(f"Corrupt or incomplete WAV file '{file_path}': {exc}") from exc

    return audio, sample_rate


def frame_audio(
    audio: np.ndarray,
    config: VADConfig,
    pad_last: bool = True,
) -> Iterator[AudioFrame]:
    """Slice a continuous 1D audio buffer into fixed-duration AudioFrames.

    Args:
        audio: 1D numpy array of audio samples (float32).
        config: VADConfig containing sample_rate, frame_duration_ms, and hop_duration_ms.
        pad_last: If True, zero-pad the last frame if it is shorter than frame_size.

    Yields:
        AudioFrame instances sequentially.
    """
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim > 1:
        raise ValueError(f"Audio must be a 1D array, got shape {audio.shape}")

    frame_size = config.frame_size
    hop_size = config.hop_size
    sample_rate = config.sample_rate
    total_samples = len(audio)

    frame_index = 0
    start_sample = 0

    while start_sample < total_samples:
        end_sample = start_sample + frame_size

        if end_sample <= total_samples:
            frame_data = audio[start_sample:end_sample]
        else:
            if not pad_last:
                break
            # Zero-pad remaining tail
            padding = np.zeros(end_sample - total_samples, dtype=np.float32)
            frame_data = np.concatenate([audio[start_sample:], padding])

        t_start = start_sample / sample_rate
        t_end = (start_sample + frame_size) / sample_rate

        yield AudioFrame(
            data=frame_data,
            timestamp_start_s=t_start,
            timestamp_end_s=t_end,
            frame_index=frame_index,
            sample_rate=sample_rate,
        )

        frame_index += 1
        start_sample += hop_size
