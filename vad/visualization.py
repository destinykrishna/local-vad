"""Visualization module for audio waveforms and VAD speech detections."""

import os
from typing import List, Optional
import numpy as np

from vad.detector import SpeechSegment, FrameDecision


def render_ascii_timeline(
    duration_s: float,
    segments: List[SpeechSegment],
    width: int = 60,
) -> str:
    """Render a text-based timeline visualizer for terminal display.

    Args:
        duration_s: Total audio duration in seconds.
        segments: List of detected SpeechSegments.
        width: Character width of the timeline bar.

    Returns:
        Formatted multi-line ASCII timeline string.
    """
    if duration_s <= 0:
        return "[Empty audio]"

    bar = ["."] * width
    for seg in segments:
        idx_start = int(np.clip(round((seg.start_s / duration_s) * (width - 1)), 0, width - 1))
        idx_end = int(np.clip(round((seg.end_s / duration_s) * (width - 1)), 0, width - 1))
        for i in range(idx_start, idx_end + 1):
            bar[i] = "S"

    bar_str = "".join(bar)
    header = f"0.0s {' ' * (width - 10)} {duration_s:.2f}s"
    timeline = f"[{bar_str}]"
    legend = "Legend: [S] = Detected Speech, [.] = Silence"

    return f"{header}\n{timeline}\n{legend}"


def plot_vad_visualization(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[SpeechSegment],
    decisions: Optional[List[FrameDecision]] = None,
    output_path: str = "vad_plot.png",
    title: Optional[str] = None,
) -> Optional[str]:
    """Plot waveform, detected speech intervals, and frame-level metrics using matplotlib.

    Args:
        audio: 1D numpy array of audio samples (float32).
        sample_rate: Sample rate in Hz.
        segments: List of detected speech segments.
        decisions: Optional list of FrameDecisions with frame-level features and scores.
        output_path: File path to save the generated PNG plot.
        title: Plot title.

    Returns:
        Path to the saved PNG image, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    duration_s = len(audio) / sample_rate
    time_axis = np.linspace(0, duration_s, len(audio), endpoint=False)

    # Subsample waveform if very dense (> 50,000 samples) for fast, clean rendering
    step = max(1, len(audio) // 40000)
    t_sub = time_axis[::step]
    audio_sub = audio[::step]

    num_panels = 3 if (decisions and len(decisions) > 0) else 2
    fig, axes = plt.subplots(num_panels, 1, figsize=(12, 3 * num_panels), sharex=True)
    if num_panels == 2:
        ax_wave, ax_state = axes
        ax_energy = None
    else:
        ax_wave, ax_energy, ax_state = axes

    # 1. Waveform Panel
    ax_wave.plot(t_sub, audio_sub, color="#4a90e2", linewidth=0.7, alpha=0.85, label="Waveform")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim([-1.05, 1.05])
    ax_wave.grid(True, linestyle="--", alpha=0.4)

    # Shade detected speech intervals
    for idx, seg in enumerate(segments):
        lbl = "Detected Speech" if idx == 0 else ""
        ax_wave.axvspan(seg.start_s, seg.end_s, color="#2ecc71", alpha=0.35, label=lbl)

    if title:
        ax_wave.set_title(title, fontsize=12, fontweight="bold")
    ax_wave.legend(loc="upper right", framealpha=0.85)

    # 2. Frame-level Metrics Panel (if decisions provided)
    if ax_energy is not None and decisions:
        frame_times = [d.frame.timestamp_start_s for d in decisions]
        energies = [d.features.energy_db for d in decisions]
        noise_floors = [d.noise_level_db for d in decisions]
        thresholds = [d.threshold_db for d in decisions]
        scores = [d.speech_score for d in decisions]

        ax_energy.plot(frame_times, energies, color="#e67e22", label="Frame Energy (dB)", linewidth=1.2)
        ax_energy.plot(frame_times, thresholds, color="#e74c3c", linestyle="--", label="Adaptive Threshold (dB)", linewidth=1.2)
        ax_energy.plot(frame_times, noise_floors, color="#95a5a6", linestyle=":", label="Noise Floor (dB)", linewidth=1.0)
        ax_energy.set_ylabel("Energy (dB FS)")
        ax_energy.set_ylim([max(-90.0, min(energies + [-70.0]) - 5), max(energies + [-20.0]) + 5])
        ax_energy.grid(True, linestyle="--", alpha=0.4)

        # Twin axis for speech score in [0, 1]
        ax_score = ax_energy.twinx()
        ax_score.plot(frame_times, scores, color="#9b59b6", alpha=0.7, linewidth=1.0, label="Speech Score")
        ax_score.set_ylabel("Score [0, 1]", color="#9b59b6")
        ax_score.set_ylim([-0.05, 1.05])
        ax_score.tick_params(axis="y", labelcolor="#9b59b6")

        lines1, labels1 = ax_energy.get_legend_handles_labels()
        lines2, labels2 = ax_score.get_legend_handles_labels()
        ax_energy.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.85)

    # 3. Binary VAD State Panel
    vad_curve = np.zeros_like(t_sub)
    for seg in segments:
        mask = (t_sub >= seg.start_s) & (t_sub <= seg.end_s)
        vad_curve[mask] = 1.0

    ax_state.plot(t_sub, vad_curve, color="#27ae60", linewidth=1.5, drawstyle="steps-post", label="VAD State (Speech=1)")
    ax_state.fill_between(t_sub, 0, vad_curve, color="#2ecc71", alpha=0.3)
    ax_state.set_ylabel("VAD State")
    ax_state.set_yticks([0, 1])
    ax_state.set_yticklabels(["SILENCE", "SPEECH"])
    ax_state.set_ylim([-0.1, 1.2])
    ax_state.set_xlabel("Time (seconds)")
    ax_state.grid(True, linestyle="--", alpha=0.4)
    ax_state.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
