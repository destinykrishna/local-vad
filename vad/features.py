"""Feature extraction module for Voice Activity Detection.

Extracts time-domain and frequency-domain features:
- Root Mean Square (RMS) energy & log energy (dB FS)
- Zero-Crossing Rate (ZCR)
- FFT-based spectral features (Spectral Centroid, Spectral Flatness, Spectral Energy, Speech Band Energy Ratio)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from vad.audio import AudioFrame
from vad.config import VADConfig


@dataclass
class FrameFeatures:
    """Acoustic feature representation for a single audio frame.

    Attributes:
        rms_energy: Linear root mean square energy.
        energy_db: Logarithmic energy in decibels relative to full scale (dB FS).
        zero_crossing_rate: Normalized rate of sign changes in time domain [0.0, 1.0].
        spectral_centroid: Center of mass of the magnitude spectrum in Hz.
        spectral_flatness: Ratio of geometric mean to arithmetic mean of power spectrum [0.0, 1.0].
        spectral_energy: Total energy in the spectrum.
        speech_band_energy_ratio: Fraction of spectral energy in speech band (e.g. 250 - 4000 Hz).
    """

    rms_energy: float = 0.0
    energy_db: float = -100.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flatness: float = 1.0
    spectral_energy: float = 0.0
    speech_band_energy_ratio: float = 0.0


class FeatureExtractor:
    """Extracts acoustic features from an AudioFrame.

    Keeps feature extraction purely acoustic and independent from state machine logic.
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        """Initialize FeatureExtractor with optional VAD configuration."""
        self.config = config or VADConfig()
        self._window_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def get_window(self, window_type: str, length: int) -> np.ndarray:
        """Retrieve or create a cached window of specified type and length.

        Args:
            window_type: Type of window ('hamming', 'hann', 'blackman', 'rectangular').
            length: Length of window in samples.

        Returns:
            1D numpy float32 array representing window weights.
        """
        key = (window_type.lower(), length)
        if key in self._window_cache:
            return self._window_cache[key]

        w_name = window_type.lower()
        if w_name == "hamming":
            win = np.hamming(length).astype(np.float32)
        elif w_name in ("hann", "hanning"):
            win = np.hanning(length).astype(np.float32)
        elif w_name == "blackman":
            win = np.blackman(length).astype(np.float32)
        elif w_name in ("rectangular", "boxcar", "none"):
            win = np.ones(length, dtype=np.float32)
        else:
            win = np.hamming(length).astype(np.float32)

        self._window_cache[key] = win
        return win

    def extract(self, frame: AudioFrame) -> FrameFeatures:
        """Extract all acoustic features from an AudioFrame.

        Args:
            frame: AudioFrame containing raw audio samples and metadata.

        Returns:
            FrameFeatures containing computed metrics.
        """
        signal = frame.data
        if len(signal) == 0:
            return FrameFeatures()

        rms = self.compute_rms(signal)
        energy_db = self.compute_energy_db(rms)
        zcr = self.compute_zcr(signal)
        centroid, flatness, spec_energy, band_ratio = self.compute_spectral_features(
            signal, frame.sample_rate
        )

        return FrameFeatures(
            rms_energy=rms,
            energy_db=energy_db,
            zero_crossing_rate=zcr,
            spectral_centroid=centroid,
            spectral_flatness=flatness,
            spectral_energy=spec_energy,
            speech_band_energy_ratio=band_ratio,
        )

    def compute_rms(self, signal: np.ndarray) -> float:
        """Compute Root Mean Square (RMS) energy of a signal frame.

        Args:
            signal: 1D numpy array of float32 samples.

        Returns:
            RMS energy (float).
        """
        if len(signal) == 0:
            return 0.0
        mean_sq = float(np.mean(signal.astype(np.float64) ** 2))
        return float(np.sqrt(mean_sq))

    def compute_energy_db(
        self, rms: float, eps: float = 1e-7, min_db: float = -100.0
    ) -> float:
        """Convert linear RMS energy to decibels relative to full scale (dB FS).

        Args:
            rms: Linear RMS amplitude.
            eps: Small epsilon to prevent log of zero.
            min_db: Minimum dB floor clamping value.

        Returns:
            Energy in dB (float).
        """
        if rms <= 0.0:
            return float(min_db)
        db = 20.0 * np.log10(max(rms, eps))
        return float(max(min_db, db))

    def compute_zcr(self, signal: np.ndarray) -> float:
        """Compute normalized zero-crossing rate of a signal frame.

        Args:
            signal: 1D numpy array of float32 samples.

        Returns:
            Normalized zero-crossing rate between 0.0 and 1.0.
        """
        if len(signal) < 2:
            return 0.0
        signs = np.signbit(signal)
        crossings = np.count_nonzero(signs[1:] != signs[:-1])
        return float(crossings / (len(signal) - 1))

    def compute_spectral_features(
        self, signal: np.ndarray, sample_rate: int
    ) -> Tuple[float, float, float, float]:
        """Compute FFT-based spectral features: centroid, flatness, spectral energy, and speech band ratio.

        Window is applied before the FFT to reduce spectral leakage.

        Args:
            signal: 1D numpy array of audio samples.
            sample_rate: Sampling frequency in Hz.

        Returns:
            Tuple of (spectral_centroid, spectral_flatness, spectral_energy, speech_band_energy_ratio).
        """
        n_samples = len(signal)
        if n_samples == 0 or sample_rate <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        # Apply analysis window
        window = self.get_window(self.config.window_type, n_samples)
        windowed = signal * window

        # Real FFT
        fft_complex = np.fft.rfft(windowed)
        mag = np.abs(fft_complex)
        power = (mag ** 2) / n_samples
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        total_spectral_energy = float(np.sum(power))
        mag_sum = float(np.sum(mag))

        # Check for near-silent signal
        if mag_sum < 1e-12 or total_spectral_energy < 1e-12:
            return (0.0, 0.0, 0.0, 0.0)

        # 1. Spectral Centroid (center of mass)
        centroid = float(np.sum(freqs * mag) / mag_sum)

        # 2. Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean)
        eps = 1e-12
        log_power = np.log(power + eps)
        geom_mean = np.exp(np.mean(log_power))
        arith_mean = np.mean(power) + eps
        flatness = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))

        # 3. Speech Band Energy Ratio (energy in e.g. 250 - 4000 Hz / total energy)
        band_mask = (freqs >= self.config.spectral_centroid_min_hz) & (
            freqs <= self.config.spectral_centroid_max_hz
        )
        band_energy = float(np.sum(power[band_mask]))
        band_ratio = float(
            np.clip(band_energy / (total_spectral_energy + eps), 0.0, 1.0)
        )

        return (centroid, flatness, total_spectral_energy, band_ratio)
