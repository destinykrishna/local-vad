"""Tests for audio loading, validation, edge cases, framing, and timestamps."""

import os
import tempfile
import wave
import pytest
import numpy as np

from vad.config import VADConfig
from vad.audio import AudioFrame, load_wav, frame_audio


def create_temp_wav(
    samples: np.ndarray,
    sample_rate: int = 16000,
    channels: int = 1,
    sampwidth: int = 2,
) -> str:
    """Helper to create a temporary WAV file with specific parameters."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        if len(samples) > 0:
            if sampwidth == 2:
                raw = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            elif sampwidth == 1:
                raw = (np.clip(samples, -1.0, 1.0) * 127.0 + 128.0).astype(np.uint8).tobytes()
            elif sampwidth == 4:
                raw = (np.clip(samples, -1.0, 1.0) * 2147483647.0).astype(np.int32).tobytes()
            else:
                raw = b""
            wf.writeframes(raw)
        else:
            wf.writeframes(b"")

    return temp_file.name


class TestAudioLoadingAndValidation:
    def test_load_valid_mono_16k(self):
        # 0.5s of 440 Hz tone
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_path = create_temp_wav(tone, sample_rate=sr, channels=1, sampwidth=2)

        try:
            audio, loaded_sr = load_wav(wav_path, target_sr=16000)
            assert loaded_sr == 16000
            assert len(audio) == len(tone)
            assert audio.dtype == np.float32
            # Check maximum amplitude closely matches 0.5
            assert np.max(audio) == pytest.approx(0.5, abs=0.02)
        finally:
            os.remove(wav_path)

    def test_load_empty_wav(self):
        wav_path = create_temp_wav(np.empty(0, dtype=np.float32), sample_rate=16000, channels=1)
        try:
            audio, loaded_sr = load_wav(wav_path, target_sr=16000)
            assert loaded_sr == 16000
            assert len(audio) == 0
            assert audio.dtype == np.float32
        finally:
            os.remove(wav_path)

    def test_load_very_short_wav(self):
        # 10 samples (shorter than standard 20ms frame of 320 samples)
        short_data = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.0, 0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        wav_path = create_temp_wav(short_data, sample_rate=16000, channels=1)
        try:
            audio, loaded_sr = load_wav(wav_path, target_sr=16000)
            assert len(audio) == 10
            assert loaded_sr == 16000
        finally:
            os.remove(wav_path)

    def test_reject_multichannel_wav(self):
        tone = np.zeros(1600, dtype=np.float32)
        wav_path = create_temp_wav(tone, sample_rate=16000, channels=2)
        try:
            with pytest.raises(ValueError, match="Expected mono WAV file"):
                load_wav(wav_path, target_sr=16000)
        finally:
            os.remove(wav_path)

    def test_reject_wrong_sample_rate(self):
        tone = np.zeros(8000, dtype=np.float32)
        wav_path = create_temp_wav(tone, sample_rate=8000, channels=1)
        try:
            with pytest.raises(ValueError, match="Expected sample rate 16000 Hz"):
                load_wav(wav_path, target_sr=16000)
        finally:
            os.remove(wav_path)

    def test_non_existent_file(self):
        with pytest.raises(FileNotFoundError):
            load_wav("this_file_does_not_exist_xyz.wav")

    def test_corrupted_or_non_wav_file(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.write(b"NOT_A_VALID_WAV_HEADER_DATA_123456789")
        temp_file.close()
        try:
            with pytest.raises(ValueError, match="Invalid or corrupt WAV file"):
                load_wav(temp_file.name)
        finally:
            os.remove(temp_file.name)


class TestFramingAndTimestamps:
    def test_frame_sizing_default_20ms(self):
        config = VADConfig(sample_rate=16000, frame_duration_ms=20.0, hop_duration_ms=20.0)
        assert config.frame_size == 320
        assert config.hop_size == 320

        # Exactly 1 second of audio (16,000 samples)
        audio = np.zeros(16000, dtype=np.float32)
        frames = list(frame_audio(audio, config))
        assert len(frames) == 50
        for i, f in enumerate(frames):
            assert f.num_samples == 320
            assert f.frame_index == i
            assert f.sample_rate == 16000

    def test_frame_sizing_custom_duration(self):
        # 10 ms frames, 10 ms hop -> 160 samples per frame
        config = VADConfig(sample_rate=16000, frame_duration_ms=10.0, hop_duration_ms=10.0)
        assert config.frame_size == 160
        audio = np.zeros(1600, dtype=np.float32)  # 100 ms
        frames = list(frame_audio(audio, config))
        assert len(frames) == 10

    def test_frame_accurate_timestamps(self):
        config = VADConfig(sample_rate=16000, frame_duration_ms=20.0, hop_duration_ms=20.0)
        audio = np.zeros(16000, dtype=np.float32)
        frames = list(frame_audio(audio, config))

        for idx, frame in enumerate(frames):
            expected_start = idx * 0.02
            expected_end = (idx + 1) * 0.02
            assert frame.timestamp_start_s == pytest.approx(expected_start, abs=1e-6)
            assert frame.timestamp_end_s == pytest.approx(expected_end, abs=1e-6)
            assert frame.duration_s == pytest.approx(0.02, abs=1e-6)

    def test_framing_with_padding_short_signal(self):
        config = VADConfig(sample_rate=16000, frame_duration_ms=20.0, hop_duration_ms=20.0)
        # 100 samples < 320 frame_size
        audio = np.ones(100, dtype=np.float32)
        frames = list(frame_audio(audio, config, pad_last=True))
        assert len(frames) == 1
        assert frames[0].num_samples == 320
        # First 100 should be 1.0, remaining 220 should be 0.0
        np.testing.assert_array_equal(frames[0].data[:100], np.ones(100, dtype=np.float32))
        np.testing.assert_array_equal(frames[0].data[100:], np.zeros(220, dtype=np.float32))

    def test_framing_without_padding_short_signal(self):
        config = VADConfig(sample_rate=16000, frame_duration_ms=20.0, hop_duration_ms=20.0)
        audio = np.ones(100, dtype=np.float32)
        frames = list(frame_audio(audio, config, pad_last=False))
        assert len(frames) == 0

    def test_framing_empty_audio(self):
        config = VADConfig()
        audio = np.empty(0, dtype=np.float32)
        frames = list(frame_audio(audio, config))
        assert len(frames) == 0

    def test_framing_invalid_dimensions(self):
        config = VADConfig()
        two_d_audio = np.zeros((2, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="Audio must be a 1D array"):
            list(frame_audio(two_d_audio, config))
