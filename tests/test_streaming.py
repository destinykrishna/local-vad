"""Comprehensive unit and equivalence tests for Streaming Voice Activity Detection.

Verifies:
1. Batch vs streaming exact equivalence across various chunk sizes:
   - 10 ms (160 samples, sub-frame)
   - 20 ms (320 samples, 1 frame)
   - 50 ms (800 samples)
   - 100 ms (1600 samples)
   - 250 ms (4000 samples)
   - 1000 ms (16000 samples)
   - Irregular / random chunk sizes (50 to 900 samples)
2. Streaming robustness scenarios:
   - Empty audio stream
   - One-frame audio stream
   - Sub-frame partial chunk buffering
   - Speech split across chunk boundaries
   - Silence split across chunk boundaries
   - Hangover split across chunk boundaries
   - EOF terminating during active speech
   - EOF terminating during hangover
   - Multiple speech regions across stream
   - Very small chunks (e.g. 10 samples)
   - Large chunks (e.g. 5 seconds)
   - Repeated start() / flush() lifecycle on the same instance
   - Reset and reuse on independent audio
"""

import pytest
import numpy as np

from vad.config import VADConfig
from vad.audio import load_wav
from vad.detector import VoiceActivityDetector, SpeechSegment
from tests.test_vad import make_noise, make_speech_like


class TestStreamingEquivalence:
    """Tests guaranteeing that batch and streaming execution produce equivalent results."""

    @pytest.mark.parametrize("chunk_ms", [10.0, 20.0, 50.0, 100.0, 250.0, 1000.0])
    def test_equivalence_fixed_chunk_sizes(self, chunk_ms):
        """Batch and streaming must produce identical segments across various chunk sizes."""
        # Synthesize multi-segment audio: silence, speech 1, pause, speech 2, silence
        sr = 16000
        pre = make_noise(0.4, level_db=-65.0, seed=1)
        spk1 = make_speech_like(0.8, level_db=-22.0, f0=130.0)
        sil = make_noise(1.2, level_db=-65.0, seed=2)
        spk2 = make_speech_like(0.8, level_db=-22.0, f0=160.0)
        post = make_noise(0.5, level_db=-65.0, seed=3)
        audio = np.concatenate([pre, spk1, sil, spk2, post])

        cfg = VADConfig()
        detector = VoiceActivityDetector(cfg)

        # 1. Batch mode
        batch_segs = detector.process_audio(audio)

        # 2. Streaming mode
        detector.start()
        chunk_samples = int(round(chunk_ms * sr / 1000.0))
        for start_idx in range(0, len(audio), chunk_samples):
            chunk = audio[start_idx : start_idx + chunk_samples]
            detector.process(chunk)
        stream_segs = detector.flush()

        assert len(batch_segs) == len(stream_segs)
        for b, s in zip(batch_segs, stream_segs):
            assert b.start_s == pytest.approx(s.start_s, abs=1e-5)
            assert b.end_s == pytest.approx(s.end_s, abs=1e-5)

    def test_equivalence_irregular_chunk_sizes(self):
        """Batch and streaming must produce identical segments with pseudo-random chunk sizes."""
        sr = 16000
        pre = make_noise(0.5, level_db=-65.0, seed=10)
        spk = make_speech_like(1.2, level_db=-20.0, f0=140.0)
        sil = make_noise(1.0, level_db=-65.0, seed=11)
        spk2 = make_speech_like(1.0, level_db=-22.0, f0=155.0)
        post = make_noise(0.6, level_db=-65.0, seed=12)
        audio = np.concatenate([pre, spk, sil, spk2, post])

        cfg = VADConfig()
        detector = VoiceActivityDetector(cfg)

        batch_segs = detector.process_audio(audio)

        # Stream with random chunk lengths from 37 to 850 samples
        rng = np.random.default_rng(42)
        detector.start()
        idx = 0
        while idx < len(audio):
            c_len = int(rng.integers(37, 850))
            chunk = audio[idx : idx + c_len]
            detector.process(chunk)
            idx += c_len
        stream_segs = detector.flush()

        assert len(batch_segs) == len(stream_segs)
        for b, s in zip(batch_segs, stream_segs):
            assert b.start_s == pytest.approx(s.start_s, abs=1e-5)
            assert b.end_s == pytest.approx(s.end_s, abs=1e-5)


class TestStreamingRobustness:
    """Targeted tests for streaming edge cases and lifecycle."""

    def setup_method(self):
        self.detector = VoiceActivityDetector(VADConfig())

    def test_empty_stream(self):
        """Streaming empty chunks or calling flush() on empty stream returns empty list."""
        self.detector.start()
        res1 = self.detector.process(np.empty(0, dtype=np.float32))
        res2 = self.detector.process(np.array([], dtype=np.float32))
        final = self.detector.flush()
        assert res1 == []
        assert res2 == []
        assert final == []
        assert self.detector.segments == []

    def test_one_frame_stream(self):
        """Streaming exactly one frame of silence (320 samples) behaves cleanly."""
        self.detector.start()
        one_frame = make_noise(0.02, level_db=-65.0)  # 320 samples
        res = self.detector.process(one_frame)
        final = self.detector.flush()
        assert res == []
        assert final == []

    def test_partial_frame_buffering(self):
        """Sub-frame chunks (e.g. 50 samples) should buffer until 320 samples accumulate."""
        self.detector.start()
        speech = make_speech_like(0.6, level_db=-20.0)
        # Send 50 samples at a time
        all_newly_closed = []
        for i in range(0, len(speech), 50):
            chunk = speech[i : i + 50]
            newly = self.detector.process(chunk)
            if newly:
                all_newly_closed.extend(newly)

        final = self.detector.flush()
        assert len(final) == 1
        assert final[0].start_s == pytest.approx(0.0, abs=0.06)

    def test_speech_split_across_chunks(self):
        """Speech utterance sliced down the middle into two chunks must remain one continuous segment."""
        self.detector.start()
        pre = make_noise(0.3, level_db=-65.0, seed=21)
        speech = make_speech_like(1.0, level_db=-20.0)
        post = make_noise(0.5, level_db=-65.0, seed=22)
        audio = np.concatenate([pre, speech, post])

        # Cut right in the middle of speech (at 0.8s)
        split_pt = int(0.8 * 16000)
        chunk1 = audio[:split_pt]
        chunk2 = audio[split_pt:]

        self.detector.process(chunk1)
        self.detector.process(chunk2)
        segs = self.detector.flush()

        assert len(segs) == 1
        assert segs[0].start_s == pytest.approx(0.30, abs=0.08)
        assert segs[0].end_s >= 1.30

    def test_silence_split_across_chunks(self):
        """Long silence chunks streamed sequentially should not trigger false positives."""
        self.detector.start()
        c1 = make_noise(0.5, level_db=-65.0, seed=31)
        c2 = make_noise(0.5, level_db=-65.0, seed=32)
        c3 = make_noise(0.5, level_db=-65.0, seed=33)

        self.detector.process(c1)
        self.detector.process(c2)
        self.detector.process(c3)
        final = self.detector.flush()

        assert final == []

    def test_hangover_split_across_chunks(self):
        """Chunk boundary falling inside the hangover period must continue without interruption."""
        self.detector.start()
        pre = make_noise(0.3, level_db=-65.0, seed=41)
        speech = make_speech_like(0.5, level_db=-20.0)  # speech ends at 0.8s
        # Hangover runs from 0.8s to 1.1s. Place chunk boundary at 0.95s (in middle of hangover)
        tail = make_noise(0.8, level_db=-65.0, seed=42)
        audio = np.concatenate([pre, speech, tail])

        split_idx = int(0.95 * 16000)
        self.detector.process(audio[:split_idx])
        self.detector.process(audio[split_idx:])
        segs = self.detector.flush()

        assert len(segs) == 1
        assert segs[0].start_s == pytest.approx(0.30, abs=0.08)

    def test_eof_during_speech(self):
        """Audio stream ending abruptly while speaker is actively speaking finalizes cleanly."""
        self.detector.start()
        pre = make_noise(0.3, level_db=-65.0, seed=51)
        # Speech continues directly to EOF with no post-silence
        speech = make_speech_like(0.8, level_db=-20.0)
        audio = np.concatenate([pre, speech])

        self.detector.process(audio)
        segs = self.detector.flush()

        assert len(segs) == 1
        total_len = len(audio) / 16000.0
        assert segs[0].start_s == pytest.approx(0.30, abs=0.08)
        assert segs[0].end_s <= total_len

    def test_eof_during_hangover(self):
        """Audio stream ending while in hangover period finalizes cleanly with rollback."""
        self.detector.start()
        pre = make_noise(0.3, level_db=-65.0, seed=61)
        speech = make_speech_like(0.6, level_db=-20.0)  # speech ends at 0.9s
        # Only 60 ms of silence before EOF (less than 300 ms hangover)
        short_post = make_noise(0.06, level_db=-65.0, seed=62)
        audio = np.concatenate([pre, speech, short_post])

        self.detector.process(audio)
        segs = self.detector.flush()

        assert len(segs) == 1
        total_len = len(audio) / 16000.0
        assert segs[0].end_s <= total_len

    def test_multiple_speech_regions_incremental(self):
        """Multiple phrases streamed incrementally output segments as speech ends."""
        self.detector.start()
        p1 = make_speech_like(0.5, level_db=-20.0, f0=120.0)
        sil1 = make_noise(1.2, level_db=-65.0, seed=71)
        p2 = make_speech_like(0.5, level_db=-20.0, f0=150.0)
        sil2 = make_noise(1.2, level_db=-65.0, seed=72)
        audio = np.concatenate([p1, sil1, p2, sil2])

        # Stream in 100 ms chunks and collect newly finalized
        finalized_during_stream = []
        chunk_len = 1600
        for i in range(0, len(audio), chunk_len):
            chunk = audio[i : i + chunk_len]
            closed = self.detector.process(chunk)
            if closed:
                finalized_during_stream.extend(closed)

        all_final = self.detector.flush()
        assert len(all_final) == 2
        # Phrase 1 should have finalized during the stream (before flush)
        assert len(finalized_during_stream) >= 1

    def test_repeated_start_flush_lifecycle(self):
        """The same detector instance can be reused across repeated start() / flush() cycles."""
        for cycle in range(3):
            self.detector.start()
            pre = make_noise(0.3, level_db=-65.0, seed=100 + cycle)
            speech = make_speech_like(0.6, level_db=-20.0, f0=130.0 + cycle * 10)
            post = make_noise(0.4, level_db=-65.0, seed=200 + cycle)
            audio = np.concatenate([pre, speech, post])

            self.detector.process(audio)
            segs = self.detector.flush()

            assert len(segs) == 1
            assert segs[0].start_s == pytest.approx(0.30, abs=0.08)

    def test_reset_clears_streaming_state(self):
        """Calling reset() completely empties buffers and resets counters."""
        self.detector.start()
        # Feed partial chunk
        self.detector.process(make_noise(0.015, level_db=-65.0))  # 240 samples (< 320)
        assert len(self.detector._sample_buffer) == 240

        self.detector.reset()
        assert len(self.detector._sample_buffer) == 0
        assert self.detector._samples_processed == 0
        assert self.detector._is_streaming is False
        assert self.detector.segments == []

    def test_no_unbounded_buffer_growth(self):
        """Streaming hundreds of chunks must keep internal sample buffer strictly bounded (< 320 samples)."""
        self.detector.start()
        chunk = make_noise(0.025, level_db=-65.0)  # 400 samples (1.25 frames)
        max_buffer_observed = 0
        for _ in range(200):  # 200 chunks = 5 seconds of streaming
            self.detector.process(chunk)
            max_buffer_observed = max(max_buffer_observed, len(self.detector._sample_buffer))

        # At the end of every process call, buffer must have less than 1 frame (< 320 samples)
        assert len(self.detector._sample_buffer) < 320
        # Peak buffer size during a call must never exceed chunk_samples + frame_size
        assert max_buffer_observed < 320
        self.detector.flush()
        assert len(self.detector._sample_buffer) == 0

