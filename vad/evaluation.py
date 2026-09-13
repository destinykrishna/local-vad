"""Evaluation module for comparing VAD detections against ground-truth speech intervals.

Calculates:
- Ground-truth speech duration
- Detected speech duration
- True positive, missed speech, and false positive durations
- Precision, Recall, F1 score, and Temporal IoU (Jaccard)
- Onset and offset boundary timing errors
- Segment counts and fragmentation / merging diagnostic observations
- Per-recording and aggregate metrics across entire audio datasets
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from vad.detector import SpeechSegment


@dataclass
class FragmentationMergingStats:
    """Observations on segmentation fragmentation, merging, and false triggers.

    Attributes:
        fragmented_gt_segments: GT segments broken into 2+ detected segments (over-segmentation).
        merged_det_segments: Detected segments bridging across 2+ GT segments (under-segmentation).
        clean_one_to_one_matches: Segments that matched 1-to-1 cleanly.
        unmatched_false_alarms: Detected segments with zero GT overlap.
        completely_missed_gt: GT segments with zero detected overlap.
        notes: Human-readable observations summary.
    """

    fragmented_gt_segments: int = 0
    merged_det_segments: int = 0
    clean_one_to_one_matches: int = 0
    unmatched_false_alarms: int = 0
    completely_missed_gt: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert observations to dictionary."""
        return {
            "fragmented_gt_segments": self.fragmented_gt_segments,
            "merged_det_segments": self.merged_det_segments,
            "clean_one_to_one_matches": self.clean_one_to_one_matches,
            "unmatched_false_alarms": self.unmatched_false_alarms,
            "completely_missed_gt": self.completely_missed_gt,
            "notes": self.notes,
        }


@dataclass
class EvaluationMetrics:
    """Detailed VAD performance metrics for a single recording or aggregate dataset.

    Attributes:
        file_name: Name of evaluated file (or 'AGGREGATE').
        total_audio_duration_s: Audio duration in seconds.
        ground_truth_speech_s: Total ground-truth speech duration (s).
        detected_speech_s: Total detected speech duration (s).
        true_positive_speech_s: Time where both GT and VAD agree on speech (s).
        missed_speech_duration_s: GT speech missed by VAD (False Negative, s).
        false_positive_duration_s: VAD speech during silence (False Alarm, s).
        precision: Proportion of detected speech that was true speech [0.0, 1.0].
        recall: Proportion of true speech that was detected [0.0, 1.0].
        f1_score: Harmonic mean of precision and recall [0.0, 1.0].
        temporal_iou: Intersection over Union of speech time [0.0, 1.0].
        mean_onset_error_s: Mean absolute speech onset boundary timing error (s).
        mean_offset_error_s: Mean absolute speech offset boundary timing error (s).
        num_ground_truth_segments: Number of ground-truth intervals.
        num_detected_segments: Number of detected intervals.
        fragmentation: Diagnostic fragmentation and merging observations.
    """

    file_name: str
    total_audio_duration_s: float
    ground_truth_speech_s: float
    detected_speech_s: float
    true_positive_speech_s: float
    missed_speech_duration_s: float
    false_positive_duration_s: float
    precision: float
    recall: float
    f1_score: float
    temporal_iou: float
    mean_onset_error_s: float
    mean_offset_error_s: float
    num_ground_truth_segments: int
    num_detected_segments: int
    fragmentation: FragmentationMergingStats = field(default_factory=FragmentationMergingStats)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "file_name": self.file_name,
            "total_audio_duration_s": round(self.total_audio_duration_s, 3),
            "ground_truth_speech_s": round(self.ground_truth_speech_s, 3),
            "detected_speech_s": round(self.detected_speech_s, 3),
            "true_positive_speech_s": round(self.true_positive_speech_s, 3),
            "missed_speech_duration_s": round(self.missed_speech_duration_s, 3),
            "false_positive_duration_s": round(self.false_positive_duration_s, 3),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "temporal_iou": round(self.temporal_iou, 4),
            "mean_onset_error_s": round(self.mean_onset_error_s, 4),
            "mean_offset_error_s": round(self.mean_offset_error_s, 4),
            "num_ground_truth_segments": self.num_ground_truth_segments,
            "num_detected_segments": self.num_detected_segments,
            "fragmentation_observations": self.fragmentation.to_dict(),
        }

    def format_table(self) -> str:
        """Format metrics into a clean text table."""
        title = f"VAD EVALUATION METRICS: {self.file_name}"
        sep = "=" * len(title)
        lines = [
            sep,
            title,
            sep,
            f"{'Metric':<34} | {'Value'}",
            "-" * 55,
            f"{'Total Audio Duration':<34} | {self.total_audio_duration_s:.2f} s",
            f"{'Ground-Truth Speech Duration':<34} | {self.ground_truth_speech_s:.2f} s",
            f"{'Detected Speech Duration':<34} | {self.detected_speech_s:.2f} s",
            f"{'True Positive Duration (Overlap)':<34} | {self.true_positive_speech_s:.2f} s",
            f"{'Missed Speech Duration (FN)':<34} | {self.missed_speech_duration_s:.2f} s",
            f"{'False-Positive Duration (FP)':<34} | {self.false_positive_duration_s:.2f} s",
            f"{'Temporal IoU (Jaccard Index)':<34} | {self.temporal_iou * 100:.1f} %",
            f"{'Precision':<34} | {self.precision * 100:.1f} %",
            f"{'Recall (Speech Coverage)':<34} | {self.recall * 100:.1f} %",
            f"{'F1 Score':<34} | {self.f1_score * 100:.1f} %",
            f"{'Mean Onset Boundary Error':<34} | {self.mean_onset_error_s * 1000:.1f} ms",
            f"{'Mean Offset Boundary Error':<34} | {self.mean_offset_error_s * 1000:.1f} ms",
            f"{'Segment Counts (Detected / GT)':<34} | {self.num_detected_segments} / {self.num_ground_truth_segments}",
            "-" * 55,
            "Segmentation Observations:",
            f"  - Fragmented GT Segments (Over-seg) : {self.fragmentation.fragmented_gt_segments}",
            f"  - Merged Det Segments (Under-seg)   : {self.fragmentation.merged_det_segments}",
            f"  - Clean 1-to-1 Matches              : {self.fragmentation.clean_one_to_one_matches}",
            f"  - Pure False Alarm Segments         : {self.fragmentation.unmatched_false_alarms}",
            f"  - Completely Missed GT Segments     : {self.fragmentation.completely_missed_gt}",
        ]
        if self.fragmentation.notes:
            lines.append("  - Notes:")
            for note in self.fragmentation.notes:
                lines.append(f"    * {note}")

        return "\n".join(lines)


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort, validate, and merge overlapping or adjacent intervals."""
    valid_ivs = []
    for s, e in intervals:
        start = float(min(s, e))
        end = float(max(s, e))
        if end > start:
            valid_ivs.append((start, end))

    if not valid_ivs:
        return []

    sorted_ivs = sorted(valid_ivs, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_ivs[0]]

    for current in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        if current[0] <= prev_end:
            merged[-1] = (prev_start, max(prev_end, current[1]))
        else:
            merged.append(current)

    return merged


def compute_overlap(iv1: Tuple[float, float], iv2: Tuple[float, float]) -> float:
    """Compute overlapping time between two intervals in seconds."""
    return max(0.0, min(iv1[1], iv2[1]) - max(iv1[0], iv2[0]))


def load_dataset_ground_truth(gt_source: Union[str, Dict[str, Any]]) -> Dict[str, List[Tuple[float, float]]]:
    """Load ground-truth mapping for multiple audio files.

    Supported formats:
    - Path to JSON file mapping file_name -> {"segments": [{"start": x, "end": y}]}
    - Path to JSON file mapping file_name -> [{"start": x, "end": y}]
    - In-memory dictionary with same structure

    Returns:
        Dict mapping normalized file_name to list of (start_s, end_s) tuples.
    """
    if isinstance(gt_source, str):
        if not os.path.isfile(gt_source):
            raise FileNotFoundError(f"Ground-truth file not found: {gt_source}")
        with open(gt_source, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(gt_source, dict):
        data = gt_source
    else:
        raise ValueError(f"Unsupported ground-truth source type: {type(gt_source)}")

    result: Dict[str, List[Tuple[float, float]]] = {}

    for key, value in data.items():
        if key.startswith("_"):
            continue  # Ignore metadata/instruction comments

        raw_list: List[Tuple[float, float]] = []
        if isinstance(value, dict) and "segments" in value:
            val_segments = value["segments"]
        elif isinstance(value, list):
            val_segments = value
        else:
            val_segments = []

        for item in val_segments:
            if isinstance(item, dict):
                s = float(item.get("start", 0.0))
                e = float(item.get("end", 0.0))
                if e > s:
                    raw_list.append((s, e))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                s = float(item[0])
                e = float(item[1])
                if e > s:
                    raw_list.append((s, e))

        result[key] = merge_intervals(raw_list)

    return result


def load_ground_truth(source: Any) -> List[Tuple[float, float]]:
    """Legacy helper to load ground truth for a single recording."""
    if isinstance(source, dict) and "segments" in source:
        source = source["segments"]
    elif isinstance(source, str) and os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("[") or content.startswith("{"):
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "segments" in parsed:
                source = parsed["segments"]
            elif isinstance(parsed, list):
                source = parsed
            elif isinstance(parsed, dict):
                # Check if it's a dataset GT file
                first_val = next(iter(parsed.values()))
                if isinstance(first_val, dict) and "segments" in first_val:
                    source = first_val["segments"]
                else:
                    source = []

    raw_ivs: List[Tuple[float, float]] = []
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                raw_ivs.append((float(item.get("start", 0.0)), float(item.get("end", 0.0))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_ivs.append((float(item[0]), float(item[1])))

    return merge_intervals(raw_ivs)


def evaluate_detection(
    detected: Union[List[SpeechSegment], List[dict], List[Tuple[float, float]]],
    ground_truth: Union[str, List[dict], List[Tuple[float, float]]],
    total_audio_duration_s: float,
    file_name: str = "recording",
) -> EvaluationMetrics:
    """Evaluate detected speech segments against ground-truth intervals for one recording.

    Uses time-overlap based evaluation rather than requiring exact segment boundaries.
    """
    if isinstance(ground_truth, (str, dict, list)):
        gt_intervals = load_ground_truth(ground_truth)
    else:
        gt_intervals = []

    # Normalize detected segments into (start, end) tuples
    det_raw: List[Tuple[float, float]] = []
    for item in detected:
        if isinstance(item, SpeechSegment):
            det_raw.append((item.start_s, item.end_s))
        elif isinstance(item, dict):
            det_raw.append((float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            det_raw.append((float(item[0]), float(item[1])))
    det_intervals = merge_intervals(det_raw)

    gt_total_s = sum(e - s for s, e in gt_intervals)
    det_total_s = sum(e - s for s, e in det_intervals)

    # Compute total True Positive duration (temporal intersection)
    tp_total_s = 0.0
    for d_iv in det_intervals:
        for g_iv in gt_intervals:
            tp_total_s += compute_overlap(d_iv, g_iv)

    missed_s = max(0.0, gt_total_s - tp_total_s)
    false_pos_s = max(0.0, det_total_s - tp_total_s)

    union_s = gt_total_s + det_total_s - tp_total_s
    iou = (tp_total_s / union_s) if union_s > 0 else (1.0 if (gt_total_s == 0 and det_total_s == 0) else 0.0)

    precision = (tp_total_s / det_total_s) if det_total_s > 0 else (1.0 if gt_total_s == 0 else 0.0)
    recall = (tp_total_s / gt_total_s) if gt_total_s > 0 else (1.0 if det_total_s == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Boundary analysis and Fragmentation / Merging tracking
    onset_errors: List[float] = []
    offset_errors: List[float] = []
    frag_stats = FragmentationMergingStats()

    # Track GT -> Detected overlaps
    gt_to_det_matches: Dict[int, List[int]] = {g_idx: [] for g_idx in range(len(gt_intervals))}
    det_to_gt_matches: Dict[int, List[int]] = {d_idx: [] for d_idx in range(len(det_intervals))}

    for d_idx, d_iv in enumerate(det_intervals):
        for g_idx, g_iv in enumerate(gt_intervals):
            if compute_overlap(d_iv, g_iv) > 0:
                gt_to_det_matches[g_idx].append(d_idx)
                det_to_gt_matches[d_idx].append(g_idx)

    # Analyze GT segments
    for g_idx, g_iv in enumerate(gt_intervals):
        matched_dets = gt_to_det_matches[g_idx]
        if not matched_dets:
            frag_stats.completely_missed_gt += 1
            frag_stats.notes.append(f"GT segment [{g_iv[0]:.2f}s - {g_iv[1]:.2f}s] was completely missed.")
        elif len(matched_dets) > 1:
            frag_stats.fragmented_gt_segments += 1
            frag_stats.notes.append(
                f"GT segment [{g_iv[0]:.2f}s - {g_iv[1]:.2f}s] was fragmented into {len(matched_dets)} detected segments."
            )
            # Boundary error against earliest start and latest end
            earliest_start = min(det_intervals[d][0] for d in matched_dets)
            latest_end = max(det_intervals[d][1] for d in matched_dets)
            onset_errors.append(abs(earliest_start - g_iv[0]))
            offset_errors.append(abs(latest_end - g_iv[1]))
        else:
            # Exactly 1 detected segment overlaps
            d_idx = matched_dets[0]
            if len(det_to_gt_matches[d_idx]) == 1:
                frag_stats.clean_one_to_one_matches += 1
            d_iv = det_intervals[d_idx]
            onset_errors.append(abs(d_iv[0] - g_iv[0]))
            offset_errors.append(abs(d_iv[1] - g_iv[1]))

    # Analyze Detected segments for merging and false alarms
    for d_idx, d_iv in enumerate(det_intervals):
        matched_gts = det_to_gt_matches[d_idx]
        if not matched_gts:
            frag_stats.unmatched_false_alarms += 1
            frag_stats.notes.append(f"Detected segment [{d_iv[0]:.2f}s - {d_iv[1]:.2f}s] is a pure false alarm.")
        elif len(matched_gts) > 1:
            frag_stats.merged_det_segments += 1
            frag_stats.notes.append(
                f"Detected segment [{d_iv[0]:.2f}s - {d_iv[1]:.2f}s] bridges across {len(matched_gts)} GT segments (under-segmentation)."
            )

    mean_onset = float(sum(onset_errors) / len(onset_errors)) if onset_errors else 0.0
    mean_offset = float(sum(offset_errors) / len(offset_errors)) if offset_errors else 0.0

    return EvaluationMetrics(
        file_name=file_name,
        total_audio_duration_s=total_audio_duration_s,
        ground_truth_speech_s=gt_total_s,
        detected_speech_s=det_total_s,
        true_positive_speech_s=tp_total_s,
        missed_speech_duration_s=missed_s,
        false_positive_duration_s=false_pos_s,
        precision=precision,
        recall=recall,
        f1_score=f1,
        temporal_iou=iou,
        mean_onset_error_s=mean_onset,
        mean_offset_error_s=mean_offset,
        num_ground_truth_segments=len(gt_intervals),
        num_detected_segments=len(det_intervals),
        fragmentation=frag_stats,
    )


def evaluate_dataset(
    detections_by_file: Dict[str, Union[List[SpeechSegment], List[dict]]],
    ground_truth_source: Union[str, Dict[str, Any]],
    durations_by_file: Dict[str, float],
) -> Tuple[Dict[str, EvaluationMetrics], EvaluationMetrics]:
    """Evaluate full dataset of recordings, producing per-recording and aggregate metrics.

    Args:
        detections_by_file: Mapping of filename to detected segments.
        ground_truth_source: Path to ground_truth.json or loaded dictionary.
        durations_by_file: Mapping of filename to audio duration in seconds.

    Returns:
        Tuple of (per_file_metrics_dict, aggregate_metrics).
    """
    gt_map = load_dataset_ground_truth(ground_truth_source)
    per_file: Dict[str, EvaluationMetrics] = {}

    total_audio_s = 0.0
    total_gt_s = 0.0
    total_det_s = 0.0
    total_tp_s = 0.0
    all_onset_errors: List[float] = []
    all_offset_errors: List[float] = []
    total_num_gt_segs = 0
    total_num_det_segs = 0

    agg_frag = FragmentationMergingStats()

    # Match files by basename
    normalized_gt_map = {os.path.basename(k): v for k, v in gt_map.items()}

    for file_key, det_segs in detections_by_file.items():
        base_name = os.path.basename(file_key)
        dur_s = durations_by_file.get(file_key, durations_by_file.get(base_name, 0.0))
        gt_segs = normalized_gt_map.get(base_name, [])

        eval_res = evaluate_detection(
            detected=det_segs,
            ground_truth=gt_segs,
            total_audio_duration_s=dur_s,
            file_name=base_name,
        )
        per_file[base_name] = eval_res

        total_audio_s += eval_res.total_audio_duration_s
        total_gt_s += eval_res.ground_truth_speech_s
        total_det_s += eval_res.detected_speech_s
        total_tp_s += eval_res.true_positive_speech_s
        total_num_gt_segs += eval_res.num_ground_truth_segments
        total_num_det_segs += eval_res.num_detected_segments

        agg_frag.fragmented_gt_segments += eval_res.fragmentation.fragmented_gt_segments
        agg_frag.merged_det_segments += eval_res.fragmentation.merged_det_segments
        agg_frag.clean_one_to_one_matches += eval_res.fragmentation.clean_one_to_one_matches
        agg_frag.unmatched_false_alarms += eval_res.fragmentation.unmatched_false_alarms
        agg_frag.completely_missed_gt += eval_res.fragmentation.completely_missed_gt

        if eval_res.mean_onset_error_s > 0:
            all_onset_errors.append(eval_res.mean_onset_error_s)
        if eval_res.mean_offset_error_s > 0:
            all_offset_errors.append(eval_res.mean_offset_error_s)

    # Micro-averaged aggregate metrics
    agg_missed_s = max(0.0, total_gt_s - total_tp_s)
    agg_false_pos_s = max(0.0, total_det_s - total_tp_s)
    agg_union_s = total_gt_s + total_det_s - total_tp_s
    agg_iou = (total_tp_s / agg_union_s) if agg_union_s > 0 else (1.0 if (total_gt_s == 0 and total_det_s == 0) else 0.0)
    agg_prec = (total_tp_s / total_det_s) if total_det_s > 0 else (1.0 if total_gt_s == 0 else 0.0)
    agg_rec = (total_tp_s / total_gt_s) if total_gt_s > 0 else (1.0 if total_det_s == 0 else 0.0)
    agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) > 0 else 0.0

    mean_onset = float(sum(all_onset_errors) / len(all_onset_errors)) if all_onset_errors else 0.0
    mean_offset = float(sum(all_offset_errors) / len(all_offset_errors)) if all_offset_errors else 0.0

    aggregate = EvaluationMetrics(
        file_name="AGGREGATE (All Recordings)",
        total_audio_duration_s=total_audio_s,
        ground_truth_speech_s=total_gt_s,
        detected_speech_s=total_det_s,
        true_positive_speech_s=total_tp_s,
        missed_speech_duration_s=agg_missed_s,
        false_positive_duration_s=agg_false_pos_s,
        precision=agg_prec,
        recall=agg_rec,
        f1_score=agg_f1,
        temporal_iou=agg_iou,
        mean_onset_error_s=mean_onset,
        mean_offset_error_s=mean_offset,
        num_ground_truth_segments=total_num_gt_segs,
        num_detected_segments=total_num_det_segs,
        fragmentation=agg_frag,
    )

    return per_file, aggregate
