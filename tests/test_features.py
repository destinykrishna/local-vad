"""Focused unit tests for feature extraction: RMS, ZCR, FFT spectral metrics, windowing."""

import pytest
import numpy as np

from vad.config import VADConfig
from vad.audio import AudioFrame
from vad.features import FeatureExtractor, FrameFeatures


class TestRMSFeature:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_rms_silence(self):
        signal = np.zeros(320, dtype=np.float32)
        rms = self.extractor.compute_rms(signal)
        assert rms == 0.0
        db = self.extractor.compute_energy_db(rms)
        assert db == -100.0

    def test_rms_constant_dc(self):
        # Constant amplitude 0.5 -> RMS must be exactly 0.5
        signal = np.full(320, 0.5, dtype=np.float32)
        rms = self.extractor.compute_rms(signal)
        assert rms == pytest.approx(0.5, rel=1e-5)
        # dB = 20 * log10(0.5) ≈ -6.0206 dB
        db = self.extractor.compute_energy_db(rms)
        assert db == pytest.approx(-6.0206, abs=0.01)

    def test_rms_sine_wave(self):
        # Sine wave with amplitude A = 1.0 has theoretical RMS = 1 / sqrt(2) ≈ 0.707107
        sr = 16000
        t = np.linspace(0, 0.02, 320, endpoint=False)
        # Integer number of cycles: 400 Hz * 0.02s = 8 full cycles
        signal = np.sin(2 * np.pi * 400 * t).astype(np.float32)
        rms = self.extractor.compute_rms(signal)
        expected_rms = 1.0 / np.sqrt(2.0)
        assert rms == pytest.approx(expected_rms, rel=1e-3)
        # dB FS should be ≈ -3.01 dB
        db = self.extractor.compute_energy_db(rms)
        assert db == pytest.approx(-3.01, abs=0.02)

    def test_rms_empty_signal(self):
        signal = np.empty(0, dtype=np.float32)
        assert self.extractor.compute_rms(signal) == 0.0


class TestZeroCrossingRate:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_zcr_silence(self):
        signal = np.zeros(320, dtype=np.float32)
        assert self.extractor.compute_zcr(signal) == 0.0

    def test_zcr_constant_positive(self):
        signal = np.ones(320, dtype=np.float32)
        assert self.extractor.compute_zcr(signal) == 0.0

    def test_zcr_maximum_alternating(self):
        # Alternating [1, -1, 1, -1, ...] has sign changes on every step -> ZCR = 1.0
        signal = np.array([1.0, -1.0] * 160, dtype=np.float32)
        assert self.extractor.compute_zcr(signal) == 1.0

    def test_zcr_sine_wave_frequency(self):
        # For a sine wave of frequency f at sample rate fs, expected ZCR ≈ 2 * f / fs
        sr = 16000
        f = 1000.0  # 1 kHz
        t = np.linspace(0, 0.02, 320, endpoint=False)
        signal = np.sin(2 * np.pi * f * t).astype(np.float32)
        zcr = self.extractor.compute_zcr(signal)
        expected_zcr = 2.0 * f / sr  # 2000 / 16000 = 0.125
        assert zcr == pytest.approx(expected_zcr, abs=0.01)

    def test_zcr_very_short_signal(self):
        assert self.extractor.compute_zcr(np.array([0.5], dtype=np.float32)) == 0.0
        assert self.extractor.compute_zcr(np.empty(0, dtype=np.float32)) == 0.0


class TestSpectralFeaturesAndWindowing:
    def setup_method(self):
        self.config = VADConfig(
            sample_rate=16000,
            frame_duration_ms=20.0,
            spectral_centroid_min_hz=250.0,
            spectral_centroid_max_hz=4000.0,
            window_type="hamming",
        )
        self.extractor = FeatureExtractor(self.config)

    def test_window_caching_and_types(self):
        w_hamming = self.extractor.get_window("hamming", 320)
        assert len(w_hamming) == 320
        assert w_hamming[0] < 0.1  # Hamming window edge is ~0.08
        assert w_hamming[160] == pytest.approx(1.0, abs=0.02)  # center is ~1.0

        # Cached identity check
        assert self.extractor.get_window("hamming", 320) is w_hamming

        w_rect = self.extractor.get_window("rectangular", 320)
        np.testing.assert_array_equal(w_rect, np.ones(320, dtype=np.float32))

    def test_spectral_centroid_frequency_tracking(self):
        sr = 16000
        t = np.linspace(0, 0.02, 320, endpoint=False)

        # 1000 Hz tone
        tone_1k = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        c_1k, _, _, _ = self.extractor.compute_spectral_features(tone_1k, sr)
        assert c_1k == pytest.approx(1000.0, abs=50.0)

        # 3000 Hz tone
        tone_3k = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
        c_3k, _, _, _ = self.extractor.compute_spectral_features(tone_3k, sr)
        assert c_3k == pytest.approx(3000.0, abs=50.0)

        # Centroid of higher frequency must be higher
        assert c_3k > c_1k

    def test_spectral_flatness_tone_vs_noise(self):
        sr = 16000
        t = np.linspace(0, 0.02, 320, endpoint=False)

        # Pure tone: high tonality -> low flatness
        tone = np.sin(2 * np.pi * 500 * t).astype(np.float32)
        _, flatness_tone, _, _ = self.extractor.compute_spectral_features(tone, sr)
        assert flatness_tone < 0.15

        # White noise: uniform spectral distribution -> high flatness compared to tone
        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(320).astype(np.float32)
        _, flatness_noise, _, _ = self.extractor.compute_spectral_features(noise, sr)
        assert flatness_noise > 0.45
        assert flatness_noise > 3.0 * flatness_tone

    def test_speech_band_energy_ratio(self):
        sr = 16000
        t = np.linspace(0, 0.02, 320, endpoint=False)

        # In-band signal: 1000 Hz (inside 250 - 4000 Hz)
        in_band = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        _, _, _, ratio_in = self.extractor.compute_spectral_features(in_band, sr)
        assert ratio_in > 0.95  # Almost all energy in band

        # Out-of-band signal: 6000 Hz (above 4000 Hz)
        out_band = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
        _, _, _, ratio_out = self.extractor.compute_spectral_features(out_band, sr)
        assert ratio_out < 0.05  # Minimal energy in speech band

    def test_extract_returns_clean_representation(self):
        t = np.linspace(0, 0.02, 320, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
        frame = AudioFrame(
            data=sig,
            timestamp_start_s=0.10,
            timestamp_end_s=0.12,
            frame_index=5,
            sample_rate=16000,
        )
        feats = self.extractor.extract(frame)

        assert isinstance(feats, FrameFeatures)
        assert not np.isnan(feats.rms_energy)
        assert not np.isnan(feats.energy_db)
        assert not np.isnan(feats.zero_crossing_rate)
        assert not np.isnan(feats.spectral_centroid)
        assert not np.isnan(feats.spectral_flatness)
        assert not np.isnan(feats.spectral_energy)
        assert not np.isnan(feats.speech_band_energy_ratio)
        assert 0.0 <= feats.speech_band_energy_ratio <= 1.0
        assert 0.0 <= feats.spectral_flatness <= 1.0
        assert 0.0 <= feats.zero_crossing_rate <= 1.0

    def test_silent_frame_safety(self):
        silent_frame = AudioFrame(
            data=np.zeros(320, dtype=np.float32),
            timestamp_start_s=0.0,
            timestamp_end_s=0.02,
            frame_index=0,
            sample_rate=16000,
        )
        feats = self.extractor.extract(silent_frame)
        assert feats.rms_energy == 0.0
        assert feats.energy_db == -100.0
        assert feats.zero_crossing_rate == 0.0
        assert feats.spectral_centroid == 0.0
        assert feats.spectral_flatness == 0.0
        assert feats.spectral_energy == 0.0
        assert feats.speech_band_energy_ratio == 0.0
