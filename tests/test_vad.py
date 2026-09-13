"""Comprehensive unit tests for the Voice Activity Detector.

Tests required scenarios:
1. Pure silence
2. Speech-like signal
3. Speech with a short pause (bridged by hangover)
4. Two speech regions separated by long silence (split into two distinct segments)
5. Short impulse noise (rejected by onset confirmation)
6. Changing background noise (adaptive noise floor tracking)
7. Noise estimator frozen during active speech
8. Empty and very short audio
"""

import pytest
import numpy as np

from vad.config import VADConfig
from vad.detector import VoiceActivityDetector, SpeechSegment
from vad.state_machine import VADState


def make_noise(duration_s: float, level_db: float = -65.0, sample_rate: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate white Gaussian noise with specific RMS level in dB FS."""
    n_samples = int(duration_s * sample_rate)
    if n_samples == 0:
        return np.empty(0, dtype=np.float32)
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal(n_samples).astype(np.float32)
    rms_target = 10.0 ** (level_db / 20.0)
    current_rms = np.sqrt(np.mean(raw**2)) + 1e-12
    return (raw * (rms_target / current_rms)).astype(np.float32)


def make_speech_like(
    duration_s: float,
    level_db: float = -20.0,
    sample_rate: int = 16000,
    f0: float = 140.0,
) -> np.ndarray:
    """Synthesize a rich harmonic vowel-like signal with speech formant resonance."""
    n_samples = int(duration_s * sample_rate)
    if n_samples == 0:
        return np.empty(0, dtype=np.float32)

    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    # Sum harmonics in speech band (F0, 3*F0, 5*F0, 7*F0, etc.)
    signal = np.zeros(n_samples, dtype=np.float32)
    harmonics = [
        (140.0, 1.0),
        (280.0, 0.8),
        (420.0, 0.6),
        (700.0, 0.7),
        (1200.0, 0.5),
        (2400.0, 0.4),
    ]
    for freq, weight in harmonics:
        signal += weight * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Scale to desired dB
    rms_target = 10.0 ** (level_db / 20.0)
    current_rms = np.sqrt(np.mean(signal**2)) + 1e-12
    return (signal * (rms_target / current_rms)).astype(np.float32)


class TestVoiceActivityDetectorCore:
    def setup_method(self):
        self.config = VADConfig(
            sample_rate=16000,
            frame_duration_ms=20.0,
            energy_onset_margin_db=9.0,
            energy_offset_margin_db=3.5,
            onset_frames=3,          # 60 ms
            hangover_duration_ms=300.0,
            hangover_frames=15,      # 300 ms
        )
        self.detector = VoiceActivityDetector(self.config)

    def test_pure_silence(self):
        """Pure background noise/silence should return 0 speech segments."""
        silence = make_noise(2.0, level_db=-65.0)
        segments = self.detector.detect_segments(silence)
        assert segments == []
        assert self.detector.state_machine.state == VADState.SILENCE

    def test_speech_like_signal(self):
        """Speech burst bracketed by silence should return exactly 1 segment."""
        # 0.5s silence + 1.0s speech + 0.5s silence
        pre_silence = make_noise(0.5, level_db=-60.0, seed=1)
        speech = make_speech_like(1.0, level_db=-22.0)
        post_silence = make_noise(0.8, level_db=-60.0, seed=2)
        audio = np.concatenate([pre_silence, speech, post_silence])

        segments = self.detector.detect_segments(audio)
        assert len(segments) == 1

        seg = segments[0]
        # Speech begins at 0.5s, ends at 1.5s (+ hangover of ~0.3s -> ~1.8s)
        assert seg["start"] == pytest.approx(0.50, abs=0.08)
        assert seg["end"] >= 1.50
        assert seg["end"] <= 1.85

    def test_speech_with_short_pause(self):
        """Speech with a short pause (< hangover duration of 300ms) should NOT split the utterance."""
        # 0.4s silence
        # 0.8s speech
        # 0.16s pause (160 ms < 300 ms hangover)
        # 0.8s speech
        # 0.8s silence
        pre_silence = make_noise(0.4, level_db=-60.0, seed=10)
        speech_1 = make_speech_like(0.8, level_db=-20.0, f0=130.0)
        short_pause = make_noise(0.16, level_db=-60.0, seed=11)
        speech_2 = make_speech_like(0.8, level_db=-20.0, f0=150.0)
        post_silence = make_noise(0.8, level_db=-60.0, seed=12)

        audio = np.concatenate([pre_silence, speech_1, short_pause, speech_2, post_silence])
        segments = self.detector.detect_segments(audio)

        # Crucial requirement: hangover bridges the short pause into 1 single continuous segment!
        assert len(segments) == 1
        seg = segments[0]
        assert seg["start"] == pytest.approx(0.40, abs=0.08)
        # End should be after speech_2 ends (0.4 + 0.8 + 0.16 + 0.8 = 2.16s + hangover)
        assert seg["end"] >= 2.16

    def test_two_speech_regions_separated_by_long_silence(self):
        """Speech separated by a long pause (> hangover duration of 300ms) should yield 2 segments."""
        # 0.4s silence
        # 0.6s speech
        # 0.8s silence (> 300ms hangover)
        # 0.6s speech
        # 0.8s silence
        pre_silence = make_noise(0.4, level_db=-60.0, seed=20)
        speech_1 = make_speech_like(0.6, level_db=-20.0)
        long_silence = make_noise(0.8, level_db=-60.0, seed=21)
        speech_2 = make_speech_like(0.6, level_db=-20.0)
        post_silence = make_noise(0.8, level_db=-60.0, seed=22)

        audio = np.concatenate([pre_silence, speech_1, long_silence, speech_2, post_silence])
        segments = self.detector.detect_segments(audio)

        assert len(segments) == 2
        seg1, seg2 = segments[0], segments[1]
        assert seg1["start"] == pytest.approx(0.40, abs=0.08)
        # First segment must close before second segment starts
        assert seg1["end"] < seg2["start"]
        # Second segment starts around 0.4 + 0.6 + 0.8 = 1.80s
        assert seg2["start"] == pytest.approx(1.80, abs=0.08)

    def test_short_impulse_noise_rejection(self):
        """Short transient click (1 frame, 20 ms) must be rejected by onset confirmation."""
        # 0.5s silence + 0.02s loud click (-10 dB) + 0.5s silence
        pre_silence = make_noise(0.5, level_db=-60.0, seed=30)
        click = np.array([0.9, -0.8] * 160, dtype=np.float32)  # 320 samples (1 frame)
        post_silence = make_noise(0.5, level_db=-60.0, seed=31)

        audio = np.concatenate([pre_silence, click, post_silence])
        segments = self.detector.detect_segments(audio)

        # Click alone should NOT trigger confirmed speech segment
        assert segments == []

    def test_changing_background_noise_adaptation(self):
        """Noise floor estimate should adapt to rising background noise without false triggers."""
        # 3 seconds of background noise ramping smoothly from -65 dB to -45 dB
        sr = 16000
        n_samples = int(3.0 * sr)
        rng = np.random.default_rng(seed=40)
        raw_noise = rng.standard_normal(n_samples).astype(np.float32)

        ramp_db = np.linspace(-65.0, -45.0, n_samples)
        envelope = 10.0 ** (ramp_db / 20.0)
        noise_ramp = raw_noise * envelope

        # Pure noise ramp should NOT trigger speech
        segments = self.detector.detect_segments(noise_ramp)
        assert segments == []

        # Noise floor at the end of the ramp should be significantly higher than initial
        assert self.detector.noise_floor_db > -55.0

    def test_noise_estimator_freezes_during_speech(self):
        """Noise floor adaptation must NOT adapt aggressively while speech is active."""
        # Initialize detector on 0.2s silence (-60 dB)
        pre_silence = make_noise(0.2, level_db=-60.0, seed=50)
        # 1.5s of loud speech (-15 dB)
        loud_speech = make_speech_like(1.5, level_db=-15.0)

        # Process silence first
        self.detector.process_audio(pre_silence)
        noise_before_speech = self.detector.noise_floor_db

        # Process speech
        self.detector.process_audio(np.concatenate([pre_silence, loud_speech]))
        # While speech was active, noise floor should not jump up to speech level (-15 dB)
        assert self.detector.noise_floor_db < -45.0
        assert abs(self.detector.noise_floor_db - noise_before_speech) < 8.0

    def test_empty_and_very_short_audio(self):
        """Detector must handle empty audio and sub-frame audio without crashing."""
        assert self.detector.detect_segments(np.empty(0, dtype=np.float32)) == []
        assert self.detector.detect_segments(np.array([0.1, -0.1], dtype=np.float32)) == []

    def test_dictionary_output_format(self):
        """Verify output dictionary format strictly matches [{'start': x, 'end': y}]."""
        audio = np.concatenate([
            make_noise(0.3, level_db=-60.0),
            make_speech_like(0.8, level_db=-20.0),
            make_noise(0.6, level_db=-60.0),
        ])
        result = self.detector.detect_segments(audio)
        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert set(item.keys()) == {"start", "end"}
        assert isinstance(item["start"], float)
        assert isinstance(item["end"], float)
        assert item["start"] < item["end"]
