"""Tests for VAD project architecture, interfaces, and scaffold."""

import pytest
import numpy as np

from vad.config import VADConfig
from vad.audio import AudioFrame, frame_audio
from vad.features import FrameFeatures, FeatureExtractor
from vad.state_machine import VADState, VADStateMachine
from vad.detector import VoiceActivityDetector, SpeechSegment, FrameDecision
from vad.cli import create_parser


class TestVADConfig:
    def test_default_config(self):
        config = VADConfig()
        assert config.sample_rate == 16000
        assert config.frame_duration_ms == 20.0
        assert config.frame_size == 320  # 16000 * 0.02
        assert config.hop_size == 320
        assert config.hangover_frames == 15
        assert config.onset_frames == 3
        config.validate()

    def test_invalid_sample_rate(self):
        config = VADConfig(sample_rate=-1)
        with pytest.raises(ValueError):
            config.validate()

    def test_invalid_noise_adaptation(self):
        config = VADConfig(noise_adaptation_rate=1.5)
        with pytest.raises(ValueError):
            config.validate()


class TestAudioScaffold:
    def test_audio_frame(self):
        data = np.zeros(320, dtype=np.float32)
        frame = AudioFrame(
            data=data,
            timestamp_start_s=0.0,
            timestamp_end_s=0.02,
            frame_index=0,
            sample_rate=16000,
        )
        assert frame.duration_s == pytest.approx(0.02)
        assert frame.num_samples == 320

    def test_frame_audio_generator(self):
        config = VADConfig(sample_rate=16000, frame_duration_ms=20.0, hop_duration_ms=20.0)
        # 1 second of synthetic silence: 16000 samples -> exactly 50 frames of 320 samples
        audio = np.zeros(16000, dtype=np.float32)
        frames = list(frame_audio(audio, config))
        assert len(frames) == 50
        assert frames[0].timestamp_start_s == 0.0
        assert frames[0].timestamp_end_s == pytest.approx(0.02)
        assert frames[-1].frame_index == 49
        assert frames[-1].timestamp_end_s == pytest.approx(1.0)


class TestFeatureScaffold:
    def test_feature_extractor_interface(self):
        config = VADConfig()
        extractor = FeatureExtractor(config)
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, config.frame_size)).astype(np.float32)
        frame = AudioFrame(
            data=samples,
            timestamp_start_s=0.0,
            timestamp_end_s=0.02,
            frame_index=0,
        )
        features = extractor.extract(frame)
        assert isinstance(features, FrameFeatures)
        assert features.rms_energy > 0.0
        assert features.energy_db > -100.0
        assert 0.0 <= features.zero_crossing_rate <= 1.0


class TestStateMachineScaffold:
    def test_state_transitions_and_hangover(self):
        config = VADConfig(onset_frames=2, hangover_frames=2)
        sm = VADStateMachine(config)
        assert sm.state == VADState.SILENCE
        assert not sm.is_speech

        # Frame 1 candidate -> POSSIBLE_ONSET
        s1 = sm.step(is_speech_candidate=True)
        assert s1 == VADState.POSSIBLE_ONSET
        assert not sm.is_speech

        # Frame 2 candidate -> SPEECH
        s2 = sm.step(is_speech_candidate=True)
        assert s2 == VADState.SPEECH
        assert sm.is_speech

        # Non-candidate frame 1 -> HANGOVER
        s3 = sm.step(is_speech_candidate=False)
        assert s3 == VADState.HANGOVER
        assert sm.is_speech

        # Non-candidate frame 2 -> HANGOVER
        s4 = sm.step(is_speech_candidate=False)
        assert s4 == VADState.HANGOVER
        assert sm.is_speech

        # Non-candidate frame 3 -> SILENCE
        s5 = sm.step(is_speech_candidate=False)
        assert s5 == VADState.SILENCE
        assert not sm.is_speech

    def test_reset(self):
        sm = VADStateMachine()
        sm.step(True)
        sm.reset()
        assert sm.state == VADState.SILENCE


class TestDetectorScaffold:
    def test_detector_initialization_and_processing(self):
        detector = VoiceActivityDetector()
        assert detector.config.sample_rate == 16000

        # Run with 0.1s of silence
        silence = np.zeros(1600, dtype=np.float32)
        segments = detector.process_audio(silence, sample_rate=16000)
        assert isinstance(segments, list)

    def test_speech_segment_structure(self):
        seg = SpeechSegment(start_s=0.82, end_s=3.41)
        assert seg.duration_s == pytest.approx(2.59, abs=0.01)
        d = seg.to_dict()
        assert d == {"start": 0.82, "end": 3.41}
        assert seg["start"] == 0.82
        assert seg["end"] == 3.41


class TestCLIScaffold:
    def test_parser_options(self):
        parser = create_parser()
        args = parser.parse_args(["sample.wav", "--frame-ms", "30.0", "--hangover", "10", "--json"])
        assert args.wav_file == "sample.wav"
        assert args.frame_ms == 30.0
        assert args.hangover == 10
        assert args.json is True
