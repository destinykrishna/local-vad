"""Audio format converter utility for preparing audio files for VAD processing."""

import os
import wave
import numpy as np
from scipy.signal import resample_poly


def convert_to_mono_16k(input_path: str, output_path: str) -> None:
    """Convert an arbitrary WAV file to mono 16 kHz 16-bit PCM WAV.

    Does not modify the detector; provides an explicit utility for users to prepare recordings.

    Args:
        input_path: Path to input WAV file.
        output_path: Destination path for normalized mono 16 kHz WAV.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If input file is corrupted.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Support FLAC or non-WAV container decoding using bundled ffmpeg
    if input_path.lower().endswith(".flac"):
        try:
            import subprocess
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            subprocess.run(
                [
                    exe,
                    "-i", input_path,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y", output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as exc:
            raise ValueError(f"Failed to convert audio file '{input_path}': {exc}") from exc

    with wave.open(input_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # 1. Parse into float32 samples [-1.0, 1.0]
    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes ({sampwidth * 8}-bit).")

    # 2. Downmix to mono if multi-channel
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        mono_samples = np.mean(samples, axis=1)
    else:
        mono_samples = samples

    # 3. Resample to 16,000 Hz if necessary
    target_sr = 16000
    if sample_rate != target_sr:
        # Use polyphase filtering with integer ratio gcd reduction
        from math import gcd
        common = gcd(target_sr, sample_rate)
        up = target_sr // common
        down = sample_rate // common
        resampled = resample_poly(mono_samples, up, down).astype(np.float32)
    else:
        resampled = mono_samples

    # 4. Save to 16-bit PCM mono WAV
    int16_pcm = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(target_sr)
        out_wf.writeframes(int16_pcm.tobytes())
