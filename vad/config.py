"""Configuration module for Voice Activity Detector (VAD)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VADConfig:
    """Configuration parameters for audio processing and VAD decision logic.

    Attributes:
        sample_rate: Expected audio sample rate in Hz (default: 16000 Hz).
        frame_duration_ms: Duration of each analysis frame in ms (default: 20 ms).
        hop_duration_ms: Frame shift / hop duration in ms (default: 20 ms, no overlap).
        window_type: Windowing function for spectral analysis ('hamming', 'hann', 'rectangular').

        # Noise Floor Estimation
        initial_noise_frames: Number of initial frames to seed the background noise floor (default: 10 frames = 200 ms).
        max_initial_noise_db: Maximum allowed noise floor during initial seeding (default: -42.0 dB).
        noise_adaptation_rate: Exponential smoothing factor (alpha) during SILENCE (default: 0.05).
        min_noise_db: Absolute lower bound for noise floor in dB (default: -80.0 dB).
        max_noise_db: Absolute upper bound for noise floor in dB (default: -20.0 dB).
        energy_threshold_db: Baseline default noise floor / minimum energy in dB (default: -55.0 dB).

        # Adaptive Thresholding & Hysteresis (Energy & SNR)
        energy_onset_margin_db: Decibels above noise floor required to trigger speech onset (default: 9.0 dB).
        energy_offset_margin_db: Decibels above noise floor required to maintain speech state (default: 3.5 dB).

        # Multi-Feature Combined Speech Score Hysteresis
        onset_score_threshold: Combined speech score threshold to enter SPEECH state (default: 0.45).
        offset_score_threshold: Combined speech score threshold to stay in SPEECH state (default: 0.25).

        # Spectral & Time-Domain Settings
        spectral_flatness_threshold: Threshold for spectral flatness (tonality vs noise) (default: 0.50).
        spectral_centroid_min_hz: Minimum expected speech spectral centroid in Hz (default: 250.0 Hz).
        spectral_centroid_max_hz: Maximum expected speech spectral centroid in Hz (default: 4000.0 Hz).
        zcr_threshold: Upper limit for typical speech zero-crossing rate (default: 0.45).

        # State Machine & Hangover
        hangover_duration_ms: Duration to hold speech state across pauses (default: 300.0 ms).
        hangover_frames: Number of hangover frames (default: 15 frames = 300 ms at 20 ms/frame).
        onset_frames: Number of consecutive candidate frames needed to confirm SPEECH onset (default: 3 frames).
        min_speech_duration_ms: Minimum duration of a detected speech segment to be retained (default: 60.0 ms).
        min_silence_duration_ms: Minimum silence gap to split into distinct speech segments (default: 200.0 ms).
    """

    sample_rate: int = 16000
    frame_duration_ms: float = 20.0
    hop_duration_ms: float = 20.0
    window_type: str = "hamming"

    # Noise Floor & Adaptation
    initial_noise_frames: int = 10
    max_initial_noise_db: float = -42.0
    noise_adaptation_rate: float = 0.05
    min_noise_db: float = -80.0
    max_noise_db: float = -20.0
    energy_threshold_db: float = -55.0

    # Adaptive Threshold Margins (Energy Hysteresis)
    energy_onset_margin_db: float = 9.0
    energy_offset_margin_db: float = 3.5

    # Multi-Feature Combined Speech Score Thresholds (Score Hysteresis)
    onset_score_threshold: float = 0.45
    offset_score_threshold: float = 0.25

    # Spectral & Time-Domain Parameters
    spectral_flatness_threshold: float = 0.50
    spectral_centroid_min_hz: float = 250.0
    spectral_centroid_max_hz: float = 4000.0
    zcr_threshold: float = 0.45

    # State Machine & Hangover
    hangover_duration_ms: float = 300.0
    hangover_frames: int = 15
    onset_frames: int = 3
    min_speech_duration_ms: float = 60.0
    min_silence_duration_ms: float = 200.0

    # Endpoint Refinement & Rollback
    enable_endpoint_rollback: bool = True
    trailing_padding_ms: float = 80.0

    @property
    def frame_size(self) -> int:
        """Calculate number of audio samples in one analysis frame."""
        return int(round(self.sample_rate * (self.frame_duration_ms / 1000.0)))

    @property
    def hop_size(self) -> int:
        """Calculate number of audio samples per frame shift / hop."""
        return int(round(self.sample_rate * (self.hop_duration_ms / 1000.0)))

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If any parameter is out of valid bounds.
        """
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.frame_duration_ms <= 0:
            raise ValueError(f"frame_duration_ms must be positive, got {self.frame_duration_ms}")
        if self.hop_duration_ms <= 0:
            raise ValueError(f"hop_duration_ms must be positive, got {self.hop_duration_ms}")
        if not (0.0 <= self.noise_adaptation_rate <= 1.0):
            raise ValueError(f"noise_adaptation_rate must be in [0, 1], got {self.noise_adaptation_rate}")
        valid_windows = ("hamming", "hann", "hanning", "rectangular", "boxcar", "blackman")
        if self.window_type.lower() not in valid_windows:
            raise ValueError(f"Unsupported window_type '{self.window_type}'. Valid: {valid_windows}")
        if self.hangover_frames < 0:
            raise ValueError(f"hangover_frames must be non-negative, got {self.hangover_frames}")
        if self.onset_frames < 1:
            raise ValueError(f"onset_frames must be >= 1, got {self.onset_frames}")
        if self.energy_onset_margin_db < self.energy_offset_margin_db:
            raise ValueError(
                f"energy_onset_margin_db ({self.energy_onset_margin_db}) must be >= "
                f"energy_offset_margin_db ({self.energy_offset_margin_db}) for hysteresis."
            )
        if self.onset_score_threshold < self.offset_score_threshold:
            raise ValueError(
                f"onset_score_threshold ({self.onset_score_threshold}) must be >= "
                f"offset_score_threshold ({self.offset_score_threshold}) for hysteresis."
            )
        if self.trailing_padding_ms < 0:
            raise ValueError(f"trailing_padding_ms must be non-negative, got {self.trailing_padding_ms}")
        if self.min_noise_db >= self.max_noise_db:
            raise ValueError(
                f"min_noise_db ({self.min_noise_db}) must be strictly less than "
                f"max_noise_db ({self.max_noise_db})."
            )
        if self.min_speech_duration_ms < 0:
            raise ValueError(f"min_speech_duration_ms must be non-negative, got {self.min_speech_duration_ms}")
        if self.min_silence_duration_ms < 0:
            raise ValueError(f"min_silence_duration_ms must be non-negative, got {self.min_silence_duration_ms}")
