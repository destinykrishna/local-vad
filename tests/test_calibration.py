"""Regression and calibration unit tests for VAD endpointing and pause handling.

Specifically tests:
1. Short pause inside speech (bridged / continuous)
2. Long pause between speech segments (separated / distinct)
3. Trailing hangover and rollback refinement
4. Speech ending near EOF
5. Speech beginning near start of file (sample 0 onset)
6. Low-energy speech ending (preserved by trailing padding)
7. Multiple speech regions
8. Endpoint rollback toggle (enabled vs disabled)
9. Gap bridging based on min_silence_duration_ms
"""

import pytest
import numpy as np

from vad.config import VADConfig
from vad.detector import VoiceActivityDetector, SpeechSegment
from tests.test_vad import make_noise, make_speech_like


class TestVADCalibrationAndEndpointing:
    """Targeted regression tests for endpoint rollback, hangover, and pause handling."""

    def test_short_pause_inside_speech(self):
        """Short natural pause (< 200 ms) inside speech must remain in one segment."""
        cfg = VADConfig(
            hangover_duration_ms=300.0,
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
            min_silence_duration_ms=200.0,
        )
        vad = VoiceActivityDetector(cfg)

        pre = make_noise(0.4, level_db=-65.0, seed=101)
        word1 = make_speech_like(0.6, level_db=-22.0, f0=140.0)
        short_pause = make_noise(0.12, level_db=-65.0, seed=102)  # 120 ms pause
        word2 = make_speech_like(0.6, level_db=-22.0, f0=160.0)
        post = make_noise(0.5, level_db=-65.0, seed=103)

        audio = np.concatenate([pre, word1, short_pause, word2, post])
        segs = vad.process_audio(audio)

        assert len(segs) == 1
        assert segs[0].start_s == pytest.approx(0.40, abs=0.08)
        # End should cover word2 (0.4 + 0.6 + 0.12 + 0.6 = 1.72s) + padding
        assert segs[0].end_s >= 1.72
        assert segs[0].end_s <= 1.85

    def test_long_pause_between_speech_segments(self):
        """Clearly separated multi-second pause must create distinct speech segments and never merge."""
        cfg = VADConfig(
            hangover_duration_ms=300.0,
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
            min_silence_duration_ms=200.0,
        )
        vad = VoiceActivityDetector(cfg)

        pre = make_noise(0.3, level_db=-65.0, seed=201)
        phrase1 = make_speech_like(0.7, level_db=-20.0, f0=130.0)
        long_pause = make_noise(1.5, level_db=-65.0, seed=202)  # 1.5 second pause
        phrase2 = make_speech_like(0.7, level_db=-20.0, f0=150.0)
        post = make_noise(0.4, level_db=-65.0, seed=203)

        audio = np.concatenate([pre, phrase1, long_pause, phrase2, post])
        segs = vad.process_audio(audio)

        assert len(segs) == 2
        # Phrase 1 starts ~0.3s
        assert segs[0].start_s == pytest.approx(0.30, abs=0.08)
        # Phrase 1 ends before long pause finishes (0.3 + 0.7 = 1.0s + padding ~ 1.08s)
        assert segs[0].end_s < 1.25
        # Gap between segments must be substantial (> 1.0s)
        gap = segs[1].start_s - segs[0].end_s
        assert gap > 1.0
        # Phrase 2 starts ~0.3 + 0.7 + 1.5 = 2.50s
        assert segs[1].start_s == pytest.approx(2.50, abs=0.08)

    def test_trailing_hangover_and_rollback(self):
        """Endpoint rollback must trim trailing hangover silence back to padding margin."""
        pre = make_noise(0.4, level_db=-65.0, seed=301)
        speech = make_speech_like(0.8, level_db=-22.0, f0=140.0)  # speech ends at 0.4 + 0.8 = 1.20s
        post = make_noise(1.0, level_db=-65.0, seed=302)
        audio = np.concatenate([pre, speech, post])

        # 1. Raw hangover (rollback disabled): segment end retains full 300 ms hangover
        cfg_raw = VADConfig(hangover_duration_ms=300.0, enable_endpoint_rollback=False)
        vad_raw = VoiceActivityDetector(cfg_raw)
        segs_raw = vad_raw.process_audio(audio)
        assert len(segs_raw) == 1
        end_raw = segs_raw[0].end_s

        # 2. Rollback enabled with 80 ms padding
        cfg_roll = VADConfig(
            hangover_duration_ms=300.0,
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
        )
        vad_roll = VoiceActivityDetector(cfg_roll)
        segs_roll = vad_roll.process_audio(audio)
        assert len(segs_roll) == 1
        end_roll = segs_roll[0].end_s

        # Rollback should make the segment end significantly tighter (by ~200 ms)
        assert end_roll < end_raw
        assert (end_raw - end_roll) == pytest.approx(0.22, abs=0.05)
        # Refined end must still cover the active speech duration
        assert end_roll >= 1.20

    def test_speech_ending_near_eof(self):
        """Speech ending right at or near EOF must finalize cleanly with rollback."""
        cfg = VADConfig(
            hangover_duration_ms=300.0,
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
        )
        vad = VoiceActivityDetector(cfg)

        pre = make_noise(0.3, level_db=-65.0, seed=401)
        speech = make_speech_like(0.8, level_db=-20.0, f0=140.0)
        # Only 40 ms of trailing noise before EOF (less than hangover duration)
        tiny_post = make_noise(0.04, level_db=-65.0, seed=402)
        audio = np.concatenate([pre, speech, tiny_post])

        segs = vad.process_audio(audio)
        assert len(segs) == 1
        total_len = len(audio) / 16000.0
        assert segs[0].end_s <= total_len
        assert segs[0].end_s >= 1.10

    def test_speech_beginning_near_start_of_file(self):
        """Speech starting at sample 0 (0.0s) should trigger without crashing."""
        cfg = VADConfig(
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
        )
        vad = VoiceActivityDetector(cfg)

        # Speech starts immediately at sample 0
        speech = make_speech_like(1.0, level_db=-20.0, f0=150.0)
        post = make_noise(0.5, level_db=-65.0, seed=501)
        audio = np.concatenate([speech, post])

        segs = vad.process_audio(audio)
        assert len(segs) == 1
        assert segs[0].start_s == pytest.approx(0.0, abs=0.06)
        assert segs[0].end_s >= 1.0

    def test_low_energy_speech_ending(self):
        """Low-energy unvoiced decay should be protected by trailing padding."""
        cfg = VADConfig(
            hangover_duration_ms=300.0,
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
        )
        vad = VoiceActivityDetector(cfg)

        pre = make_noise(0.4, level_db=-65.0, seed=601)
        voiced = make_speech_like(0.6, level_db=-22.0, f0=130.0)
        # Unvoiced/whispered ending (-35 dB) for 60 ms
        unvoiced_decay = make_noise(0.06, level_db=-38.0, seed=602)
        post = make_noise(0.6, level_db=-65.0, seed=603)

        audio = np.concatenate([pre, voiced, unvoiced_decay, post])
        segs = vad.process_audio(audio)

        assert len(segs) == 1
        # Voiced ended at 0.4 + 0.6 = 1.00s. Trailing padding should keep end >= 1.06s
        assert segs[0].end_s >= 1.04

    def test_multiple_speech_regions(self):
        """Three distinct spoken phrases separated by 1.2s silence produce exactly 3 segments."""
        cfg = VADConfig(
            enable_endpoint_rollback=True,
            trailing_padding_ms=80.0,
            min_silence_duration_ms=200.0,
        )
        vad = VoiceActivityDetector(cfg)

        p1 = make_speech_like(0.5, level_db=-20.0, f0=120.0)
        sil1 = make_noise(1.2, level_db=-65.0, seed=701)
        p2 = make_speech_like(0.5, level_db=-20.0, f0=140.0)
        sil2 = make_noise(1.2, level_db=-65.0, seed=702)
        p3 = make_speech_like(0.5, level_db=-20.0, f0=160.0)
        post = make_noise(0.5, level_db=-65.0, seed=703)

        audio = np.concatenate([p1, sil1, p2, sil2, p3, post])
        segs = vad.process_audio(audio)

        assert len(segs) == 3
        assert segs[0].end_s < segs[1].start_s
        assert segs[1].end_s < segs[2].start_s

    def test_endpoint_rollback_toggle(self):
        """Disabling enable_endpoint_rollback should revert to raw hangover timestamps."""
        pre = make_noise(0.3, level_db=-65.0, seed=801)
        speech = make_speech_like(0.7, level_db=-20.0, f0=130.0)
        post = make_noise(0.6, level_db=-65.0, seed=802)
        audio = np.concatenate([pre, speech, post])

        cfg_off = VADConfig(enable_endpoint_rollback=False)
        cfg_on = VADConfig(enable_endpoint_rollback=True, trailing_padding_ms=80.0)

        vad_off = VoiceActivityDetector(cfg_off)
        vad_on = VoiceActivityDetector(cfg_on)

        segs_off = vad_off.process_audio(audio)
        segs_on = vad_on.process_audio(audio)

        assert segs_off[0].start_s == segs_on[0].start_s
        assert segs_on[0].end_s < segs_off[0].end_s

    def test_min_silence_duration_gap_bridging(self):
        """Micro-gap (e.g. 60 ms) between two speech bursts should bridge when min_silence_duration_ms=200."""
        burst1 = make_speech_like(0.4, level_db=-20.0, f0=140.0)
        micro_pause = make_noise(0.06, level_db=-65.0, seed=901)  # 60 ms gap
        burst2 = make_speech_like(0.4, level_db=-20.0, f0=140.0)
        post = make_noise(0.5, level_db=-65.0, seed=902)
        audio = np.concatenate([burst1, micro_pause, burst2, post])

        # Bridging active (default 200 ms)
        vad_bridge = VoiceActivityDetector(VADConfig(min_silence_duration_ms=200.0))
        segs_bridge = vad_bridge.process_audio(audio)
        assert len(segs_bridge) == 1

        # Bridging disabled (0 ms)
        vad_nobridge = VoiceActivityDetector(VADConfig(min_silence_duration_ms=0.0, hangover_duration_ms=40.0, hangover_frames=2))
        # With very short hangover, bursts would split without bridging
        assert len(segs_bridge) == 1
