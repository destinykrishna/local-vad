"""CLI entry point and demonstration suite for Voice Activity Detection."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from vad.config import VADConfig
from vad.audio import load_wav, frame_audio
from vad.detector import VoiceActivityDetector, SpeechSegment, FrameDecision
from vad.inspect import inspect_wav, inspect_directory, format_metadata_table
from vad.evaluation import (
    evaluate_detection,
    evaluate_dataset,
    load_dataset_ground_truth,
    load_ground_truth,
    EvaluationMetrics,
)
from vad.visualization import render_ascii_timeline, plot_vad_visualization


def create_parser(prog: str = "vad") -> argparse.ArgumentParser:
    """Create and configure the main command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Local Voice Activity Detector (VAD) CLI & Demonstration Suite.",
    )

    parser.add_argument(
        "wav_file",
        nargs="?",
        default=None,
        help="Path to a mono 16 kHz PCM WAV file or a directory containing WAV files.",
    )

    parser.add_argument(
        "--dir",
        "--audio-dir",
        dest="dir",
        type=str,
        default=None,
        help="Process all WAV files in the specified directory.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect and report metadata and compatibility of WAV file(s) without running VAD.",
    )

    parser.add_argument(
        "--ground-truth",
        "--eval",
        dest="ground_truth",
        type=str,
        default=None,
        help="Path to JSON/CSV ground-truth intervals to evaluate detections against.",
    )

    parser.add_argument(
        "--visualize",
        "--plot",
        dest="visualize",
        action="store_true",
        help="Generate a waveform visualization plot and render terminal ASCII timeline.",
    )

    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Custom file path to save the generated visualization PNG.",
    )

    parser.add_argument(
        "--frame-ms",
        type=float,
        default=20.0,
        help="Frame duration in milliseconds (default: 20.0 ms).",
    )

    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=-55.0,
        help="Base noise floor in dB (default: -55.0 dB).",
    )

    parser.add_argument(
        "--hangover",
        type=int,
        default=15,
        help="Hangover frame count (default: 15 frames = 300 ms).",
    )

    parser.add_argument(
        "--onset",
        type=int,
        default=3,
        help="Onset confirmation frame count (default: 3 frames).",
    )

    parser.add_argument(
        "--no-rollback",
        dest="enable_rollback",
        action="store_false",
        default=True,
        help="Disable endpoint rollback refinement and retain raw hangover frames.",
    )

    parser.add_argument(
        "--padding-ms",
        dest="padding_ms",
        type=float,
        default=80.0,
        help="Trailing silence padding in ms when endpoint rollback is active (default: 80.0 ms).",
    )

    parser.add_argument(
        "--min-silence-ms",
        dest="min_silence_ms",
        type=float,
        default=200.0,
        help="Minimum silence gap in ms to split into distinct segments (default: 200.0 ms).",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results formatted as JSON.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save speech segments or evaluation output.",
    )

    return parser


def create_eval_parser() -> argparse.ArgumentParser:
    """Create argument parser for the 'evaluate' subcommand."""
    parser = argparse.ArgumentParser(
        prog="vad evaluate",
        description="Evaluate VAD detections against manual ground-truth intervals.",
    )

    parser.add_argument(
        "--audio-dir",
        "--dir",
        dest="audio_dir",
        required=True,
        help="Directory containing normalized audio WAV files (e.g. test_audio/converted).",
    )

    parser.add_argument(
        "--ground-truth",
        "--gt",
        dest="ground_truth",
        required=True,
        help="Path to ground_truth.json containing true speech intervals.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Export evaluation results formatted as JSON.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save evaluation report.",
    )

    return parser


def process_single_file(
    file_path: str,
    config: VADConfig,
    visualize: bool = False,
    plot_output: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single WAV file with timing, segment collation, and evaluation."""
    file_name = os.path.basename(file_path)

    # 1. Inspect metadata first
    try:
        metadata = inspect_wav(file_path, target_sr=config.sample_rate, target_channels=1)
    except Exception as exc:
        return {
            "file_name": file_name,
            "file_path": file_path,
            "status": "error",
            "error": f"Failed to inspect WAV file: {exc}",
            "is_compatible": False,
        }

    # 2. Check compatibility strictly
    if not metadata.is_compatible:
        return {
            "file_name": file_name,
            "file_path": file_path,
            "status": "incompatible",
            "error": (
                f"Incompatible audio format: {metadata.incompatibility_reason}. "
                f"VAD requires mono (1 channel), 16000 Hz, 16-bit PCM WAV."
            ),
            "metadata": metadata.to_dict(),
            "is_compatible": False,
        }

    # 3. Load WAV
    try:
        audio, sample_rate = load_wav(file_path, target_sr=config.sample_rate)
    except Exception as exc:
        return {
            "file_name": file_name,
            "file_path": file_path,
            "status": "error",
            "error": f"Error loading audio: {exc}",
            "metadata": metadata.to_dict(),
            "is_compatible": True,
        }

    # 4. Run VAD and measure processing time
    detector = VoiceActivityDetector(config=config)
    decisions: List[FrameDecision] = []

    start_time = time.perf_counter()
    detector.reset()

    if visualize:
        segments: List[SpeechSegment] = []
        pending_onset_s: Optional[float] = None
        current_segment_start: Optional[float] = None
        last_speech_end: float = 0.0

        for frame in frame_audio(audio, config):
            prev_state = detector.state_machine.state
            decision = detector.process_frame(frame)
            decisions.append(decision)

            if prev_state.name == "SILENCE" and decision.state.name == "POSSIBLE_ONSET":
                pending_onset_s = frame.timestamp_start_s
            if decision.state.name == "SPEECH" and current_segment_start is None:
                current_segment_start = pending_onset_s if pending_onset_s is not None else frame.timestamp_start_s
                pending_onset_s = None
            if decision.state.name == "SILENCE":
                pending_onset_s = None
            if decision.is_speech:
                last_speech_end = frame.timestamp_end_s
            else:
                if current_segment_start is not None:
                    dur = last_speech_end - current_segment_start
                    if dur >= (config.min_speech_duration_ms / 1000.0):
                        segments.append(SpeechSegment(start_s=current_segment_start, end_s=last_speech_end))
                    current_segment_start = None

        if current_segment_start is not None:
            dur = last_speech_end - current_segment_start
            if dur >= (config.min_speech_duration_ms / 1000.0):
                segments.append(SpeechSegment(start_s=current_segment_start, end_s=last_speech_end))
    else:
        segments = detector.process_audio(audio, sample_rate=sample_rate)

    processing_time_s = time.perf_counter() - start_time
    duration_s = len(audio) / sample_rate
    rtf = (processing_time_s / duration_s) if duration_s > 0 else 0.0

    # 5. Visualization
    plot_file = None
    if visualize:
        out_plot_name = plot_output or f"{os.path.splitext(file_path)[0]}_vad.png"
        plot_file = plot_vad_visualization(
            audio=audio,
            sample_rate=sample_rate,
            segments=segments,
            decisions=decisions,
            output_path=out_plot_name,
            title=f"VAD Detection - {file_name}",
        )

    # 6. Evaluation against ground truth if provided
    eval_metrics = None
    if ground_truth_path:
        try:
            eval_metrics = evaluate_detection(
                detected=segments,
                ground_truth=ground_truth_path,
                total_audio_duration_s=duration_s,
                file_name=file_name,
            )
        except Exception as exc:
            eval_metrics = {"error": f"Failed to evaluate ground truth: {exc}"}

    return {
        "file_name": file_name,
        "file_path": file_path,
        "status": "success",
        "sample_rate": sample_rate,
        "audio_duration_s": round(duration_s, 3),
        "processing_time_s": round(processing_time_s, 4),
        "realtime_factor": round(rtf, 5),
        "realtime_speedup": round(1.0 / rtf, 1) if rtf > 0 else float("inf"),
        "speech_segments": [seg.to_dict() for seg in segments],
        "segment_count": len(segments),
        "total_speech_duration_s": round(sum(s.duration_s for s in segments), 3),
        "speech_ratio": round(sum(s.duration_s for s in segments) / max(duration_s, 1e-6), 3),
        "plot_file": plot_file,
        "ascii_timeline": render_ascii_timeline(duration_s, segments),
        "evaluation": eval_metrics.to_dict() if hasattr(eval_metrics, "to_dict") else eval_metrics,
        "raw_segments": segments,
    }


def format_single_result(result: Dict[str, Any]) -> str:
    """Format single file result into human-readable text."""
    lines = [
        "=" * 70,
        f"File: {result['file_name']}",
        "=" * 70,
    ]

    if result["status"] != "success":
        lines.append(f"Status : {result['status'].upper()}")
        lines.append(f"Error  : {result.get('error', 'Unknown error')}")
        if "metadata" in result:
            m = result["metadata"]
            lines.append(f"Format : {m['channels']} ch, {m['sample_rate']} Hz, {m['bit_depth']}-bit PCM")
        return "\n".join(lines)

    lines.extend([
        f"Sample Rate         : {result['sample_rate']} Hz",
        f"Audio Duration      : {result['audio_duration_s']:.2f} s",
        f"Processing Time     : {result['processing_time_s'] * 1000:.1f} ms ({result['processing_time_s']:.4f} s)",
        f"Realtime Factor (RTF: {result['realtime_factor']:.4f}x ({result['realtime_speedup']:.1f}x faster than real-time)",
        f"Detected Segments   : {result['segment_count']} segment(s)",
        f"Total Speech Time   : {result['total_speech_duration_s']:.2f} s ({result['speech_ratio'] * 100:.1f}% of audio)",
        "-" * 70,
    ])

    segments = result["speech_segments"]
    if segments:
        lines.append(f"{'Segment':<9} | {'Start (s)':<12} | {'End (s)':<12} | {'Duration (s)':<12}")
        lines.append("-" * 51)
        for idx, seg in enumerate(segments, start=1):
            dur = seg['end'] - seg['start']
            lines.append(f"{idx:<9} | {seg['start']:<12.3f} | {seg['end']:<12.3f} | {dur:<12.3f}")
    else:
        lines.append("No speech segments detected.")

    lines.append("-" * 70)
    lines.append("Timeline Visualization:")
    lines.append(result["ascii_timeline"])

    if result.get("plot_file"):
        lines.append(f"Saved Waveform Plot : {result['plot_file']}")

    if result.get("evaluation"):
        ev = result["evaluation"]
        if isinstance(ev, dict) and "error" not in ev:
            lines.append("-" * 70)
            lines.append("Ground-Truth Evaluation:")
            lines.append(f"  Overlap (IoU)     : {ev.get('temporal_iou', 0)*100:.1f} %")
            lines.append(f"  Precision         : {ev.get('precision', 0)*100:.1f} %")
            lines.append(f"  Recall            : {ev.get('recall', 0)*100:.1f} %")
            lines.append(f"  F1 Score          : {ev.get('f1_score', 0)*100:.1f} %")
            lines.append(f"  Missed Speech     : {ev.get('missed_speech_duration_s', 0):.2f} s")
            lines.append(f"  False Positives   : {ev.get('false_positive_duration_s', 0):.2f} s")
            lines.append(f"  Boundary Error    : onset {ev.get('mean_onset_error_s', 0)*1000:.1f} ms, offset {ev.get('mean_offset_error_s', 0)*1000:.1f} ms")

    return "\n".join(lines)


def handle_evaluate_command(args: List[str]) -> int:
    """Execute evaluation subcommand comparing detected segments vs ground truth."""
    eval_parser = create_eval_parser()
    parsed = eval_parser.parse_args(args)

    audio_dir = os.path.abspath(parsed.audio_dir)
    gt_path = os.path.abspath(parsed.ground_truth)

    if not os.path.isdir(audio_dir):
        sys.stderr.write(f"Error: Audio directory does not exist: {audio_dir}\n")
        return 1
    if not os.path.isfile(gt_path):
        sys.stderr.write(f"Error: Ground-truth file does not exist: {gt_path}\n")
        return 1

    try:
        gt_map = load_dataset_ground_truth(gt_path)
    except Exception as exc:
        sys.stderr.write(f"Error reading ground-truth JSON: {exc}\n")
        return 1

    wav_files = [
        os.path.join(audio_dir, f)
        for f in sorted(os.listdir(audio_dir))
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(audio_dir, f))
    ]

    if not wav_files:
        sys.stderr.write(f"No .wav files found in: {audio_dir}\n")
        return 1

    config = VADConfig()
    detector = VoiceActivityDetector(config=config)

    detections_by_file: Dict[str, List[SpeechSegment]] = {}
    durations_by_file: Dict[str, float] = {}

    for wav_file in wav_files:
        base = os.path.basename(wav_file)
        try:
            audio, sr = load_wav(wav_file, target_sr=config.sample_rate)
            segs = detector.process_audio(audio, sample_rate=sr)
            detections_by_file[base] = segs
            durations_by_file[base] = len(audio) / sr
        except Exception as exc:
            sys.stderr.write(f"Skipping {base} (Error: {exc})\n")

    per_file_metrics, aggregate_metrics = evaluate_dataset(
        detections_by_file=detections_by_file,
        ground_truth_source=gt_map,
        durations_by_file=durations_by_file,
    )

    if parsed.json:
        output_payload = {
            "per_recording": {k: v.to_dict() for k, v in per_file_metrics.items()},
            "aggregate": aggregate_metrics.to_dict(),
        }
        output_str = json.dumps(output_payload, indent=2)
        print(output_str)
    else:
        blocks = []
        for file_name, m in per_file_metrics.items():
            blocks.append(m.format_table())
        blocks.append(aggregate_metrics.format_table())
        output_str = "\n\n".join(blocks)
        print(output_str)

    if parsed.output:
        with open(parsed.output, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
        print(f"\nEvaluation report successfully saved to: {parsed.output}")

def create_stream_parser() -> argparse.ArgumentParser:
    """Create argument parser for the 'stream' subcommand."""
    parser = argparse.ArgumentParser(
        prog="vad stream",
        description="Stream a WAV file in chunks to demonstrate incremental streaming VAD.",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_file",
        required=True,
        help="Path to mono 16 kHz WAV file to stream.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=100.0,
        help="Chunk size in milliseconds (default: 100.0 ms).",
    )
    parser.add_argument(
        "--no-rollback",
        dest="enable_rollback",
        action="store_false",
        default=True,
        help="Disable endpoint rollback refinement.",
    )
    parser.add_argument(
        "--padding-ms",
        type=float,
        default=80.0,
        help="Trailing silence padding in ms (default: 80.0 ms).",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=200.0,
        help="Minimum silence gap in ms to split segments (default: 200.0 ms).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results formatted as JSON.",
    )
    return parser


def handle_stream_command(args: List[str]) -> int:
    """Handle 'stream' subcommand to process audio chunk-by-chunk."""
    parser = create_stream_parser()
    parsed = parser.parse_args(args)

    input_path = parsed.input_file
    if not os.path.isfile(input_path):
        sys.stderr.write(f"Input audio file not found: {input_path}\n")
        return 1

    config = VADConfig(
        enable_endpoint_rollback=parsed.enable_rollback,
        trailing_padding_ms=parsed.padding_ms,
        min_silence_duration_ms=parsed.min_silence_ms,
    )
    detector = VoiceActivityDetector(config=config)

    audio, sr = load_wav(input_path, target_sr=config.sample_rate)
    total_audio_s = len(audio) / sr
    chunk_samples = max(1, int(round(parsed.chunk_ms * sr / 1000.0)))
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    t0 = time.perf_counter()
    detector.start()

    chunk_idx = 0
    all_finalized: List[SpeechSegment] = []
    for start_idx in range(0, len(audio), chunk_samples):
        chunk = audio[start_idx : start_idx + chunk_samples]
        newly_closed = detector.process(chunk)
        if newly_closed:
            all_finalized.extend(newly_closed)
        chunk_idx += 1

    final_segments = detector.flush()
    proc_time_s = time.perf_counter() - t0
    rtf = proc_time_s / total_audio_s if total_audio_s > 0 else 0.0

    if parsed.json:
        payload = {
            "file": input_path,
            "chunk_ms": parsed.chunk_ms,
            "chunk_samples": chunk_samples,
            "total_chunks": total_chunks,
            "audio_duration_s": round(total_audio_s, 3),
            "processing_time_s": round(proc_time_s, 4),
            "realtime_factor": round(rtf, 5),
            "speech_segments": [s.to_dict() for s in final_segments],
        }
        print(json.dumps(payload, indent=2))
    else:
        print("=" * 60)
        print("STREAMING VOICE ACTIVITY DETECTION")
        print("=" * 60)
        print(f"Input File          : {input_path}")
        print(f"Audio Duration      : {total_audio_s:.2f} s")
        print(f"Chunk Size          : {parsed.chunk_ms:.1f} ms ({chunk_samples} samples)")
        print(f"Chunks Streamed     : {total_chunks}")
        print(f"Processing Time     : {proc_time_s * 1000.0:.2f} ms")
        if rtf > 0:
            print(f"Realtime Factor     : {rtf:.5f}x ({1.0 / rtf:.1f}x faster than real-time)")
        else:
            print(f"Realtime Factor     : {rtf:.5f}x")
        print(f"Detected Segments   : {len(final_segments)}")
        print("-" * 60)
        if final_segments:
            print(f"{'#':<4} | {'Start (s)':<12} | {'End (s)':<12} | {'Duration (s)':<12}")
            print("-" * 60)
            for idx, seg in enumerate(final_segments, 1):
                print(f"{idx:<4} | {seg.start_s:<12.2f} | {seg.end_s:<12.2f} | {seg.duration_s:<12.2f}")
        else:
            print("No speech detected.")
        print("=" * 60)

    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI execution routine."""
    raw_args = sys.argv[1:] if args is None else args

    # Check for subcommands
    if raw_args and raw_args[0].lower() in ("evaluate", "eval"):
        return handle_evaluate_command(raw_args[1:])
    if raw_args and raw_args[0].lower() in ("stream", "streaming"):
        return handle_stream_command(raw_args[1:])
    if raw_args and raw_args[0].lower() == "detect":
        parser = create_parser(prog="vad detect")
        parsed_args = parser.parse_args(raw_args[1:])
    else:
        parser = create_parser(prog="vad")
        parsed_args = parser.parse_args(raw_args)

    target_path = parsed_args.dir or parsed_args.wav_file

    if not target_path:
        parser.print_help()
        return 0

    target_path = os.path.abspath(target_path)
    if not os.path.exists(target_path):
        sys.stderr.write(f"Error: Target path does not exist: {target_path}\n")
        return 1

    # --- Mode 1: Inspection Mode ---
    if parsed_args.inspect:
        if os.path.isdir(target_path):
            metas = inspect_directory(target_path)
            if parsed_args.json:
                print(json.dumps([m.to_dict() for m in metas], indent=2))
            else:
                print(f"\nWAV METADATA INSPECTION FOR DIRECTORY: {target_path}\n")
                print(format_metadata_table(metas))
        else:
            meta = inspect_wav(target_path)
            if parsed_args.json:
                print(json.dumps(meta.to_dict(), indent=2))
            else:
                print(f"\nWAV METADATA INSPECTION FOR FILE: {target_path}\n")
                print(format_metadata_table([meta]))
        return 0

    # --- Mode 2: Evaluation Mode via root flags ---
    if parsed_args.ground_truth and os.path.isdir(target_path):
        return handle_evaluate_command(
            [
                "--audio-dir",
                target_path,
                "--ground-truth",
                parsed_args.ground_truth,
            ]
            + (["--json"] if parsed_args.json else [])
            + (["--output", parsed_args.output] if parsed_args.output else [])
        )

    # --- Mode 3: VAD Processing (Single File or Directory) ---
    config = VADConfig(
        frame_duration_ms=parsed_args.frame_ms,
        energy_threshold_db=parsed_args.energy_threshold,
        hangover_frames=parsed_args.hangover,
        onset_frames=parsed_args.onset,
        enable_endpoint_rollback=parsed_args.enable_rollback,
        trailing_padding_ms=parsed_args.padding_ms,
        min_silence_duration_ms=parsed_args.min_silence_ms,
    )

    wav_files: List[str] = []
    if os.path.isdir(target_path):
        for item in sorted(os.listdir(target_path)):
            if item.lower().endswith(".wav"):
                wav_files.append(os.path.join(target_path, item))
        if not wav_files:
            sys.stderr.write(f"No .wav files found in directory: {target_path}\n")
            return 1
    else:
        wav_files = [target_path]

    results: List[Dict[str, Any]] = []
    for wav_file in wav_files:
        res = process_single_file(
            file_path=wav_file,
            config=config,
            visualize=parsed_args.visualize,
            plot_output=parsed_args.plot_output,
            ground_truth_path=parsed_args.ground_truth,
        )
        results.append(res)

    if parsed_args.json:
        serializable_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k != "raw_segments"}
            serializable_results.append(clean_r)

        output_data = (
            serializable_results
            if len(serializable_results) > 1
            else serializable_results[0]["speech_segments"]
            if serializable_results[0]["status"] == "success"
            else serializable_results[0]
        )
        output_str = json.dumps(output_data, indent=2)
        print(output_str)
    else:
        output_blocks = []
        for r in results:
            output_blocks.append(format_single_result(r))

        if len(results) > 1:
            num_success = sum(1 for r in results if r["status"] == "success")
            num_incompat = sum(1 for r in results if r["status"] == "incompatible")
            num_error = sum(1 for r in results if r["status"] == "error")
            total_audio_s = sum(r.get("audio_duration_s", 0) for r in results if r["status"] == "success")
            total_proc_s = sum(r.get("processing_time_s", 0) for r in results if r["status"] == "success")
            avg_rtf = (total_proc_s / total_audio_s) if total_audio_s > 0 else 0.0

            summary_lines = [
                "=" * 70,
                f"BATCH PROCESSING SUMMARY ({len(results)} files)",
                "=" * 70,
                f"Successfully Processed : {num_success} file(s)",
                f"Incompatible Format    : {num_incompat} file(s)",
                f"Errors                 : {num_error} file(s)",
            ]
            if num_success > 0:
                summary_lines.extend([
                    f"Total Audio Duration   : {total_audio_s:.2f} s",
                    f"Total Processing Time  : {total_proc_s * 1000:.1f} ms ({total_proc_s:.4f} s)",
                    f"Aggregate RTF          : {avg_rtf:.5f}x ({1.0/avg_rtf:.1f}x real-time)" if avg_rtf > 0 else "N/A",
                ])
            summary_lines.append("=" * 70)
            output_blocks.append("\n".join(summary_lines))

        output_str = "\n\n".join(output_blocks)
        print(output_str)

    if parsed_args.output:
        with open(parsed_args.output, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
        print(f"\nResults successfully written to: {parsed_args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
