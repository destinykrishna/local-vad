"""Tests for demonstration layer: inspection, evaluation metrics, dataset evaluation, and visualization."""

import os
import tempfile
import wave
import pytest
import numpy as np

from vad.inspect import inspect_wav, inspect_directory, format_metadata_table
from vad.evaluation import (
    load_ground_truth,
    load_dataset_ground_truth,
    evaluate_detection,
    evaluate_dataset,
    merge_intervals,
)
from vad.visualization import render_ascii_timeline, plot_vad_visualization
from vad.detector import SpeechSegment
from vad.cli import main, handle_evaluate_command


def create_dummy_wav(path: str, channels: int = 1, sr: int = 16000, duration_s: float = 1.0):
    """Create a dummy WAV file."""
    n_samples = int(sr * duration_s)
    raw = (np.zeros(n_samples * channels, dtype=np.int16)).tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


class TestInspection:
    def test_inspect_compatible_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = tf.name
        try:
            create_dummy_wav(path, channels=1, sr=16000, duration_s=1.5)
            meta = inspect_wav(path)
            assert meta.is_compatible is True
            assert meta.num_channels == 1
            assert meta.sample_rate == 16000
            assert meta.bit_depth == 16
            assert meta.duration_s == pytest.approx(1.5, abs=0.01)
            assert meta.incompatibility_reason is None
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_inspect_incompatible_stereo_and_sample_rate(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = tf.name
        try:
            create_dummy_wav(path, channels=2, sr=48000, duration_s=2.0)
            meta = inspect_wav(path)
            assert meta.is_compatible is False
            assert "stereo" in meta.incompatibility_reason
            assert "48000 Hz" in meta.incompatibility_reason
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestEvaluationMetrics:
    def test_merge_intervals(self):
        ivs = [(1.0, 2.0), (1.5, 3.0), (4.0, 5.0)]
        merged = merge_intervals(ivs)
        assert merged == [(1.0, 3.0), (4.0, 5.0)]

    def test_perfect_detection_metrics(self):
        gt = [(1.0, 3.0), (5.0, 7.0)]
        det = [SpeechSegment(start_s=1.0, end_s=3.0), SpeechSegment(start_s=5.0, end_s=7.0)]
        metrics = evaluate_detection(det, gt, total_audio_duration_s=10.0, file_name="test.wav")

        assert metrics.true_positive_speech_s == pytest.approx(4.0)
        assert metrics.missed_speech_duration_s == pytest.approx(0.0)
        assert metrics.false_positive_duration_s == pytest.approx(0.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1_score == pytest.approx(1.0)
        assert metrics.temporal_iou == pytest.approx(1.0)
        assert metrics.mean_onset_error_s == pytest.approx(0.0)
        assert metrics.mean_offset_error_s == pytest.approx(0.0)
        assert metrics.fragmentation.clean_one_to_one_matches == 2

    def test_partial_overlap_and_errors(self):
        # Ground truth: [1.0, 3.0] (2.0s)
        # Detected: [1.2, 3.5] (2.3s)
        # Overlap: [1.2, 3.0] = 1.8s
        # Missed: [1.0, 1.2] = 0.2s
        # False Positive: [3.0, 3.5] = 0.5s
        gt = [{"start": 1.0, "end": 3.0}]
        det = [SpeechSegment(start_s=1.2, end_s=3.5)]
        metrics = evaluate_detection(det, gt, total_audio_duration_s=5.0, file_name="partial.wav")

        assert metrics.true_positive_speech_s == pytest.approx(1.8)
        assert metrics.missed_speech_duration_s == pytest.approx(0.2)
        assert metrics.false_positive_duration_s == pytest.approx(0.5)
        assert metrics.precision == pytest.approx(1.8 / 2.3)
        assert metrics.recall == pytest.approx(1.8 / 2.0)
        assert metrics.mean_onset_error_s == pytest.approx(0.2)
        assert metrics.mean_offset_error_s == pytest.approx(0.5)

    def test_fragmentation_over_segmentation(self):
        # One GT segment [1.0, 5.0] fragmented into two detected segments [1.0, 2.5] and [3.0, 5.0]
        gt = [(1.0, 5.0)]
        det = [(1.0, 2.5), (3.0, 5.0)]
        metrics = evaluate_detection(det, gt, total_audio_duration_s=6.0)

        assert metrics.fragmentation.fragmented_gt_segments == 1
        assert metrics.true_positive_speech_s == pytest.approx(3.5)
        assert metrics.missed_speech_duration_s == pytest.approx(0.5)  # [2.5, 3.0] was missed

    def test_merging_under_segmentation(self):
        # Two GT segments [1.0, 2.5] and [3.5, 5.0] bridged by a single detected segment [1.0, 5.0]
        gt = [(1.0, 2.5), (3.5, 5.0)]
        det = [(1.0, 5.0)]
        metrics = evaluate_detection(det, gt, total_audio_duration_s=6.0)

        assert metrics.fragmentation.merged_det_segments == 1
        assert metrics.true_positive_speech_s == pytest.approx(3.0)
        assert metrics.false_positive_duration_s == pytest.approx(1.0)  # [2.5, 3.5] false alarm bridge

    def test_dataset_level_evaluation(self):
        detections = {
            "rec1.wav": [(1.0, 3.0)],
            "rec2.wav": [(0.5, 2.5)],
        }
        gt_data = {
            "rec1.wav": {"segments": [{"start": 1.0, "end": 3.0}]},
            "rec2.wav": {"segments": [{"start": 0.5, "end": 2.5}]},
        }
        durations = {
            "rec1.wav": 5.0,
            "rec2.wav": 5.0,
        }

        per_file, agg = evaluate_dataset(detections, gt_data, durations)
        assert len(per_file) == 2
        assert agg.f1_score == pytest.approx(1.0)
        assert agg.total_audio_duration_s == pytest.approx(10.0)
        assert agg.ground_truth_speech_s == pytest.approx(4.0)
        assert agg.detected_speech_s == pytest.approx(4.0)


class TestVisualization:
    def test_render_ascii_timeline(self):
        segments = [SpeechSegment(start_s=2.0, end_s=4.0)]
        timeline = render_ascii_timeline(10.0, segments, width=50)
        assert "[S]" in timeline
        assert "0.0s" in timeline
        assert "10.00s" in timeline
        assert "S" in timeline

    def test_plot_vad_visualization_generates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            plot_path = tf.name
        try:
            audio = np.sin(np.linspace(0, 2 * np.pi * 10, 16000)).astype(np.float32)
            segments = [SpeechSegment(start_s=0.2, end_s=0.8)]
            out = plot_vad_visualization(audio, 16000, segments, output_path=plot_path)
            if out is not None:
                assert os.path.exists(plot_path)
                assert os.path.getsize(plot_path) > 0
        finally:
            if os.path.exists(plot_path):
                os.remove(plot_path)
