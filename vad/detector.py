"""Detector module for Voice Activity Detection.

Coordinates feature extraction, adaptive noise floor estimation, hysteresis-based
classification, and hangover state transitions to detect contiguous speech segments.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from vad.config import VADConfig
from vad.audio import AudioFrame, load_wav, frame_audio
from vad.features import FrameFeatures, FeatureExtractor
from vad.state_machine import VADState, VADStateMachine


@dataclass
class SpeechSegment:
    """Represents a continuous segment of detected speech with timestamps.

    Attributes:
        start_s: Start timestamp of speech in seconds.
        end_s: End timestamp of speech in seconds.
    """

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Duration of speech segment in seconds."""
        return round(self.end_s - self.start_s, 3)

    def to_dict(self, decimals: int = 2) -> dict:
        """Convert segment to dict with rounded timestamps: {'start': 0.82, 'end': 3.41}."""
        return {
            "start": round(self.start_s, decimals),
            "end": round(self.end_s, decimals),
        }

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style indexing for seamless compatibility."""
        if key == "start":
            return round(self.start_s, 2)
        elif key == "end":
            return round(self.end_s, 2)
        elif key == "duration":
            return self.duration_s
        raise KeyError(key)


@dataclass
class FrameDecision:
    """Detailed decision information for an individual analyzed frame.

    Attributes:
        frame: The audio frame processed.
        features: Computed acoustic features.
        speech_score: Combined deterministic speech likelihood score in [0.0, 1.0].
        is_candidate: Boolean indicator from hysteresis thresholding.
        is_speech: Final state machine speech decision (including hangover).
        state: State machine state (SILENCE, POSSIBLE_ONSET, SPEECH, HANGOVER).
        noise_level_db: Current estimated background noise floor in dB.
        threshold_db: Active adaptive energy threshold in dB.
    """

    frame: AudioFrame
    features: FrameFeatures
    speech_score: float
    is_candidate: bool
    is_speech: bool
    state: VADState
    noise_level_db: float
    threshold_db: float


class VoiceActivityDetector:
    """Voice Activity Detector with adaptive noise floor and hysteresis state machine."""

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        state_machine: Optional[VADStateMachine] = None,
    ) -> None:
        """Initialize the VoiceActivityDetector.

        Args:
            config: VADConfig specifying processing parameters.
            feature_extractor: Acoustic feature extraction component.
            state_machine: State transition management component.
        """
        self.config = config or VADConfig()
        self.config.validate()

        self.feature_extractor = feature_extractor or FeatureExtractor(self.config)
        self.state_machine = state_machine or VADStateMachine(self.config)

        # Dynamic state
        self.noise_floor_db: float = self.config.energy_threshold_db
        self.frames_processed: int = 0
        self._initial_energies: List[float] = []

        # Streaming session state
        self._sample_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._samples_processed: int = 0
        self._frame_index: int = 0
        self._pending_onset_start_s: Optional[float] = None
        self._current_segment_start_s: Optional[float] = None
        self._last_speech_end_s: float = 0.0
        self._last_candidate_end_s: float = 0.0
        self._pending_segment: Optional[SpeechSegment] = None
        self._completed_segments: List[SpeechSegment] = []
        self._is_streaming: bool = False

    def compute_speech_score(self, features: FrameFeatures) -> float:
        """Combine acoustic features into a deterministic speech score in [0.0, 1.0].

        Features used:
        1. SNR delta above noise floor (energy_db - noise_floor_db).
        2. Speech band energy ratio (energy in 250 - 4000 Hz relative to full spectrum).
        3. Spectral Flatness (tonality vs noise Wiener entropy).
        4. Zero-Crossing Rate (typical speech vs high-frequency noise).
        5. Absolute silence gating.

        Args:
            features: FrameFeatures extracted from current frame.

        Returns:
            Float score in [0.0, 1.0].
        """
        # Absolute silence gate
        if features.energy_db <= self.config.min_noise_db + 2.0 or features.rms_energy < 1e-6:
            return 0.0

        # 1. SNR component
        snr_delta = features.energy_db - self.noise_floor_db
        onset_m = self.config.energy_onset_margin_db
        offset_m = self.config.energy_offset_margin_db
        snr_score = float(np.clip((snr_delta - offset_m) / (onset_m - offset_m + 1e-6), 0.0, 1.0))

        # 2. Speech band energy ratio (expected > 0.40 for voice)
        band_score = float(np.clip((features.speech_band_energy_ratio - 0.30) / 0.45, 0.0, 1.0))

        # 3. Tonality score from spectral flatness (pure tone/formants ~ 0.0, noise ~ 0.6)
        tonality_score = float(np.clip((0.55 - features.spectral_flatness) / 0.40, 0.0, 1.0))

        # 4. Zero-crossing rate score (voiced speech < 0.35, high-freq noise > 0.50)
        zcr_limit = self.config.zcr_threshold
        if features.zero_crossing_rate <= zcr_limit:
            zcr_score = 1.0
        else:
            zcr_score = float(np.clip((0.65 - features.zero_crossing_rate) / 0.20, 0.0, 1.0))

        # 5. Centroid factor
        c_min = self.config.spectral_centroid_min_hz
        c_max = self.config.spectral_centroid_max_hz
        if c_min <= features.spectral_centroid <= c_max:
            centroid_factor = 1.0
        else:
            centroid_factor = 0.70

        # Weighted combination
        score = (
            0.45 * snr_score
            + 0.25 * band_score
            + 0.20 * tonality_score
            + 0.10 * zcr_score
        ) * centroid_factor

        return float(np.clip(score, 0.0, 1.0))

    def update_noise_floor(
        self, frame_energy_db: float, is_speech: bool, speech_score: float = 0.0
    ) -> None:
        """Update background noise floor estimate using recursive asymmetric tracking.

        Freezes adaptation during active speech to prevent threshold corruption.

        Args:
            frame_energy_db: Log energy of the current frame in dB.
            is_speech: Whether current frame is classified as active speech or hangover.
            speech_score: Speech likelihood score to prevent initial speech from corrupting noise floor.
        """
        # Requirement: Prevent noise estimator from adapting aggressively while speech is active
        if is_speech:
            return

        self.frames_processed += 1

        # Initial noise calibration period (only calibrate from non-speech frames)
        if self.frames_processed <= self.config.initial_noise_frames:
            # If initial frame has high energy or high speech score, do not seed noise floor with it
            if (
                frame_energy_db <= self.config.max_initial_noise_db
                and speech_score < self.config.onset_score_threshold
            ):
                self._initial_energies.append(frame_energy_db)

            if self._initial_energies:
                median_noise = float(np.median(self._initial_energies))
                self.noise_floor_db = float(
                    np.clip(median_noise, self.config.min_noise_db, self.config.max_initial_noise_db)
                )
            return

        # Continuous asymmetric adaptation during SILENCE
        if frame_energy_db < self.noise_floor_db:
            # Noise decreased: track down faster
            alpha = min(1.0, self.config.noise_adaptation_rate * 2.0)
        else:
            # Noise increased: track up slower
            alpha = self.config.noise_adaptation_rate

        self.noise_floor_db = (1.0 - alpha) * self.noise_floor_db + alpha * frame_energy_db
        self.noise_floor_db = float(
            np.clip(self.noise_floor_db, self.config.min_noise_db, self.config.max_noise_db)
        )

    def classify_frame(self, features: FrameFeatures) -> tuple[bool, float, float]:
        """Classify if frame is a speech candidate using hysteresis thresholds.

        Hysteresis:
        - In SILENCE: requires higher threshold (onset_score_threshold & energy_onset_margin_db).
        - In SPEECH: requires lower threshold (offset_score_threshold & energy_offset_margin_db).

        Args:
            features: FrameFeatures for current frame.

        Returns:
            Tuple of (is_candidate, speech_score, active_threshold_db).
        """
        speech_score = self.compute_speech_score(features)
        current_state = self.state_machine.state

        if current_state in (VADState.SPEECH, VADState.HANGOVER):
            active_energy_thresh = self.noise_floor_db + self.config.energy_offset_margin_db
            is_candidate = (
                speech_score >= self.config.offset_score_threshold
                and features.energy_db >= active_energy_thresh
            )
        else:
            active_energy_thresh = self.noise_floor_db + self.config.energy_onset_margin_db
            is_candidate = (
                speech_score >= self.config.onset_score_threshold
                and features.energy_db >= active_energy_thresh
            )

        return is_candidate, speech_score, active_energy_thresh

    def process_frame(self, frame: AudioFrame) -> FrameDecision:
        """Process a single audio frame through the full VAD pipeline.

        Args:
            frame: AudioFrame to process.

        Returns:
            FrameDecision containing feature extraction and state machine results.
        """
        features = self.feature_extractor.extract(frame)
        is_candidate, score, threshold_db = self.classify_frame(features)
        state = self.state_machine.step(is_candidate)

        # Update noise floor (only adapts during silence)
        self.update_noise_floor(
            features.energy_db,
            is_speech=self.state_machine.is_speech,
            speech_score=score,
        )

        return FrameDecision(
            frame=frame,
            features=features,
            speech_score=score,
            is_candidate=is_candidate,
            is_speech=self.state_machine.is_speech,
            state=state,
            noise_level_db=self.noise_floor_db,
            threshold_db=threshold_db,
        )

    def _finalize_segment_end(
        self,
        segment_start_s: float,
        last_speech_end_s: float,
        last_candidate_end_s: float,
    ) -> float:
        """Calculate refined segment end timestamp with optional hangover rollback.

        When speech definitively ends (hangover expires or stream ends), backtrack
        trailing non-speech frames while preserving a conservative trailing padding
        to protect unvoiced phonetic offsets.
        """
        if self.config.enable_endpoint_rollback and last_candidate_end_s > 0:
            trailing_padding_s = self.config.trailing_padding_ms / 1000.0
            refined_end = min(last_speech_end_s, last_candidate_end_s + trailing_padding_s)
            min_dur_s = self.config.min_speech_duration_ms / 1000.0
            return max(segment_start_s + min_dur_s, refined_end)
        return last_speech_end_s

    def start(self) -> None:
        """Initialize or reset an incremental streaming VAD session.

        Clears sample buffer, frame counters, and segment states so a fresh
        stream can be processed in arbitrary chunks.
        """
        self.reset()
        self._is_streaming = True

    def process(self, chunk: np.ndarray) -> List[SpeechSegment]:
        """Process an arbitrary chunk of audio samples incrementally.

        Accepts chunks of any size (smaller than one frame, multiple frames,
        or irregular lengths). Incomplete trailing samples are buffered internally
        until enough samples arrive for the next analysis frame.

        Args:
            chunk: 1D array of audio samples (float32, normalized [-1.0, 1.0]).

        Returns:
            List of SpeechSegment instances that were newly finalized during this chunk.
        """
        if not self._is_streaming:
            self.start()

        if len(chunk) == 0:
            return []

        chunk_arr = np.asarray(chunk, dtype=np.float32).squeeze()
        if chunk_arr.ndim == 0:
            chunk_arr = chunk_arr.reshape(1)
        elif chunk_arr.ndim > 1:
            raise ValueError(f"Audio chunk must be 1D, got shape {chunk_arr.shape}")

        self._sample_buffer = (
            np.concatenate([self._sample_buffer, chunk_arr])
            if len(self._sample_buffer) > 0
            else chunk_arr
        )

        frame_size = self.config.frame_size
        hop_size = self.config.hop_size
        sr = self.config.sample_rate
        newly_finalized: List[SpeechSegment] = []

        while len(self._sample_buffer) >= frame_size:
            frame_data = self._sample_buffer[:frame_size]
            t_start = self._samples_processed / sr
            t_end = (self._samples_processed + frame_size) / sr

            frame = AudioFrame(
                data=frame_data,
                timestamp_start_s=t_start,
                timestamp_end_s=t_end,
                frame_index=self._frame_index,
                sample_rate=sr,
            )

            finalized = self._step_frame(frame)
            if finalized:
                newly_finalized.extend(finalized)

            self._frame_index += 1
            self._samples_processed += hop_size
            self._sample_buffer = self._sample_buffer[hop_size:]

        return newly_finalized

    def _step_frame(self, frame: AudioFrame) -> List[SpeechSegment]:
        """Process one frame through detector, updating state and pending segments."""
        prev_state = self.state_machine.state
        decision = self.process_frame(frame)
        newly_finalized: List[SpeechSegment] = []

        # Track candidate activity for endpoint refinement
        if decision.is_candidate:
            self._last_candidate_end_s = frame.timestamp_end_s

        # 1. Onset hysteresis tracking
        if prev_state == VADState.SILENCE and decision.state == VADState.POSSIBLE_ONSET:
            self._pending_onset_start_s = frame.timestamp_start_s

        # 2. Confirmed speech onset
        if decision.state == VADState.SPEECH and self._current_segment_start_s is None:
            onset_time = (
                self._pending_onset_start_s
                if self._pending_onset_start_s is not None
                else frame.timestamp_start_s
            )
            min_gap_s = self.config.min_silence_duration_ms / 1000.0
            if (
                self._pending_segment is not None
                and self.config.min_silence_duration_ms > 0
                and (onset_time - self._pending_segment.end_s) < min_gap_s
            ):
                # Gap was smaller than min_silence_duration_ms: merge with pending segment!
                self._current_segment_start_s = self._pending_segment.start_s
                self._pending_segment = None
            else:
                if self._pending_segment is not None:
                    # Gap is >= min_silence_duration_ms: commit the pending segment
                    self._completed_segments.append(self._pending_segment)
                    newly_finalized.append(self._pending_segment)
                    self._pending_segment = None
                self._current_segment_start_s = onset_time

            self._pending_onset_start_s = None

        # 3. Discard unconfirmed onset if returning to silence
        if decision.state == VADState.SILENCE:
            self._pending_onset_start_s = None

        # 4. Track speech activity and close segment on transition to SILENCE
        if decision.is_speech:
            self._last_speech_end_s = frame.timestamp_end_s
        else:
            if self._current_segment_start_s is not None:
                seg_end = self._finalize_segment_end(
                    self._current_segment_start_s,
                    self._last_speech_end_s,
                    self._last_candidate_end_s,
                )
                duration = seg_end - self._current_segment_start_s
                if duration >= (self.config.min_speech_duration_ms / 1000.0):
                    closed_seg = SpeechSegment(
                        start_s=self._current_segment_start_s,
                        end_s=seg_end,
                    )
                    if self.config.min_silence_duration_ms > 0:
                        if self._pending_segment is not None:
                            min_gap_s = self.config.min_silence_duration_ms / 1000.0
                            if (closed_seg.start_s - self._pending_segment.end_s) < min_gap_s:
                                closed_seg = SpeechSegment(
                                    start_s=self._pending_segment.start_s,
                                    end_s=max(self._pending_segment.end_s, closed_seg.end_s),
                                )
                            else:
                                self._completed_segments.append(self._pending_segment)
                                newly_finalized.append(self._pending_segment)
                        self._pending_segment = closed_seg
                    else:
                        self._completed_segments.append(closed_seg)
                        newly_finalized.append(closed_seg)

                self._current_segment_start_s = None

        # 5. If silence has continued past min_silence_duration_ms, commit pending segment
        if self._pending_segment is not None and not decision.is_speech:
            min_gap_s = self.config.min_silence_duration_ms / 1000.0
            if (frame.timestamp_end_s - self._pending_segment.end_s) >= min_gap_s:
                self._completed_segments.append(self._pending_segment)
                newly_finalized.append(self._pending_segment)
                self._pending_segment = None

        return newly_finalized

    def flush(self) -> List[SpeechSegment]:
        """Flush any remaining buffered audio samples and finalize active speech segments.

        Returns:
            List of all SpeechSegment instances detected across the entire stream.
        """
        if not self._is_streaming:
            return list(self._completed_segments)

        frame_size = self.config.frame_size
        sr = self.config.sample_rate

        # If residual tail samples remain, zero-pad to frame_size and process final frame
        if len(self._sample_buffer) > 0:
            rem_len = len(self._sample_buffer)
            padding = np.zeros(frame_size - rem_len, dtype=np.float32)
            frame_data = np.concatenate([self._sample_buffer, padding])
            t_start = self._samples_processed / sr
            t_end = (self._samples_processed + frame_size) / sr

            frame = AudioFrame(
                data=frame_data,
                timestamp_start_s=t_start,
                timestamp_end_s=t_end,
                frame_index=self._frame_index,
                sample_rate=sr,
            )
            self._step_frame(frame)
            self._samples_processed += rem_len
            self._sample_buffer = np.empty(0, dtype=np.float32)

        # Audio ended while speaking or in hangover
        if self._current_segment_start_s is not None:
            seg_end = self._finalize_segment_end(
                self._current_segment_start_s,
                self._last_speech_end_s,
                self._last_candidate_end_s,
            )
            duration = seg_end - self._current_segment_start_s
            if duration >= (self.config.min_speech_duration_ms / 1000.0):
                closed_seg = SpeechSegment(
                    start_s=self._current_segment_start_s,
                    end_s=seg_end,
                )
                if self._pending_segment is not None and self.config.min_silence_duration_ms > 0:
                    min_gap_s = self.config.min_silence_duration_ms / 1000.0
                    if (closed_seg.start_s - self._pending_segment.end_s) < min_gap_s:
                        closed_seg = SpeechSegment(
                            start_s=self._pending_segment.start_s,
                            end_s=max(self._pending_segment.end_s, closed_seg.end_s),
                        )
                    else:
                        self._completed_segments.append(self._pending_segment)
                    self._pending_segment = None

                self._completed_segments.append(closed_seg)
            self._current_segment_start_s = None

        # Commit any remaining pending segment
        if self._pending_segment is not None:
            self._completed_segments.append(self._pending_segment)
            self._pending_segment = None

        self._is_streaming = False
        return list(self._completed_segments)

    @property
    def segments(self) -> List[SpeechSegment]:
        """All speech segments detected so far in the current session."""
        all_segs = list(self._completed_segments)
        if self._pending_segment is not None:
            all_segs.append(self._pending_segment)
        return all_segs

    def process_audio(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[SpeechSegment]:
        """Process a 1D audio array and return contiguous timestamped speech segments.

        Args:
            audio: 1D numpy array of audio samples (float32).
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of SpeechSegment instances.
        """
        self.start()
        self.process(audio)
        return self.flush()

    def detect_segments(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[dict]:
        """Process audio array and return speech segments in [{'start': x, 'end': y}] form."""
        segments = self.process_audio(audio, sample_rate=sample_rate)
        return [seg.to_dict() for seg in segments]

    def process_file(self, wav_path: str) -> List[SpeechSegment]:
        """Load a WAV file and process it through the VAD pipeline."""
        audio, sample_rate = load_wav(wav_path, target_sr=self.config.sample_rate)
        return self.process_audio(audio, sample_rate=sample_rate)

    def reset(self) -> None:
        """Reset internal states for processing a new audio stream."""
        self.state_machine.reset()
        self.noise_floor_db = self.config.energy_threshold_db
        self.frames_processed = 0
        self._initial_energies.clear()
        self._sample_buffer = np.empty(0, dtype=np.float32)
        self._samples_processed = 0
        self._frame_index = 0
        self._pending_onset_start_s = None
        self._current_segment_start_s = None
        self._last_speech_end_s = 0.0
        self._last_candidate_end_s = 0.0
        self._pending_segment = None
        self._completed_segments = []
        self._is_streaming = False
