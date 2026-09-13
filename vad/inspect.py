"""Audio inspection and metadata validation module."""

from dataclasses import dataclass
import os
from typing import List, Optional
import wave


@dataclass
class AudioMetadata:
    """Metadata container for a WAV file.

    Attributes:
        file_path: Absolute or relative path to the file.
        file_name: Base name of the file.
        num_channels: Number of audio channels (1 = mono, 2 = stereo).
        sample_rate: Sampling rate in Hz.
        sample_width_bytes: Bytes per sample (2 = 16-bit PCM).
        bit_depth: Bits per sample (8, 16, 24, 32).
        num_frames: Total number of audio frames/samples.
        duration_s: Audio duration in seconds.
        is_compatible: True if file is mono 16 kHz PCM.
        incompatibility_reason: Explanation if file is not compatible.
    """

    file_path: str
    file_name: str
    num_channels: int
    sample_rate: int
    sample_width_bytes: int
    bit_depth: int
    num_frames: int
    duration_s: float
    is_compatible: bool
    incompatibility_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary representation."""
        return {
            "file_name": self.file_name,
            "channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "num_frames": self.num_frames,
            "duration_s": round(self.duration_s, 3),
            "is_compatible": self.is_compatible,
            "incompatibility_reason": self.incompatibility_reason,
        }


def inspect_wav(file_path: str, target_sr: int = 16000, target_channels: int = 1) -> AudioMetadata:
    """Inspect and extract detailed metadata from a WAV file.

    Args:
        file_path: Path to the WAV file.
        target_sr: Required sample rate (default 16000 Hz).
        target_channels: Required channels (default 1 = mono).

    Returns:
        AudioMetadata with compatibility flag and reason.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not a valid WAV file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with wave.open(file_path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
    except wave.Error as exc:
        raise ValueError(f"Invalid or corrupt WAV file '{file_path}': {exc}") from exc
    except EOFError as exc:
        raise ValueError(f"Corrupt or incomplete WAV file '{file_path}': {exc}") from exc

    bit_depth = sample_width * 8
    duration_s = (num_frames / sample_rate) if sample_rate > 0 else 0.0

    # Check compatibility
    issues: List[str] = []
    if num_channels != target_channels:
        ch_desc = "stereo (2 channels)" if num_channels == 2 else f"{num_channels} channels"
        issues.append(f"channels={num_channels} ({ch_desc}), expected mono (1 channel)")
    if sample_rate != target_sr:
        issues.append(f"sample_rate={sample_rate} Hz, expected {target_sr} Hz")
    if sample_width != 2:
        issues.append(f"bit_depth={bit_depth}-bit, expected 16-bit PCM")

    is_compatible = len(issues) == 0
    reason = "; ".join(issues) if issues else None

    return AudioMetadata(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        num_channels=num_channels,
        sample_rate=sample_rate,
        sample_width_bytes=sample_width,
        bit_depth=bit_depth,
        num_frames=num_frames,
        duration_s=duration_s,
        is_compatible=is_compatible,
        incompatibility_reason=reason,
    )


def inspect_directory(
    dir_path: str, target_sr: int = 16000, target_channels: int = 1
) -> List[AudioMetadata]:
    """Inspect all WAV files in a given directory.

    Args:
        dir_path: Directory containing WAV files.
        target_sr: Expected sample rate (default: 16000 Hz).
        target_channels: Expected channel count (default: 1 = mono).

    Returns:
        List of AudioMetadata for each WAV found, sorted by filename.
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    results: List[AudioMetadata] = []
    for item in sorted(os.listdir(dir_path)):
        if item.lower().endswith(".wav"):
            full_path = os.path.join(dir_path, item)
            try:
                meta = inspect_wav(full_path, target_sr=target_sr, target_channels=target_channels)
                results.append(meta)
            except Exception as exc:
                results.append(
                    AudioMetadata(
                        file_path=full_path,
                        file_name=item,
                        num_channels=0,
                        sample_rate=0,
                        sample_width_bytes=0,
                        bit_depth=0,
                        num_frames=0,
                        duration_s=0.0,
                        is_compatible=False,
                        incompatibility_reason=f"Error reading file: {exc}",
                    )
                )

    return results


def format_metadata_table(metadata_list: List[AudioMetadata]) -> str:
    """Format a list of AudioMetadata into a readable summary table."""
    if not metadata_list:
        return "No WAV files found."

    header = (
        f"{'File Name':<22} | {'Channels':<9} | {'Sample Rate':<12} | "
        f"{'Bit Depth':<10} | {'Duration':<10} | {'Status'}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for m in metadata_list:
        if m.is_compatible:
            status = "OK (Compatible)"
        else:
            status = f"INCOMPATIBLE: {m.incompatibility_reason}"
        lines.append(
            f"{m.file_name:<22} | {m.num_channels:<9} | {f'{m.sample_rate} Hz':<12} | "
            f"{f'{m.bit_depth}-bit':<10} | {f'{m.duration_s:.2f}s':<10} | {status}"
        )

    return "\n".join(lines)
