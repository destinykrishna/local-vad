"""State machine module for Voice Activity Detection.

Manages speech state transitions with hysteresis and hangover logic to prevent
fragmentation of words and spurious onset triggers from transient noises.
"""

from enum import Enum
from typing import Optional

from vad.config import VADConfig


class VADState(str, Enum):
    """Enumeration of VAD states."""

    SILENCE = "SILENCE"
    POSSIBLE_ONSET = "POSSIBLE_ONSET"
    SPEECH = "SPEECH"
    HANGOVER = "HANGOVER"


class VADStateMachine:
    """State machine governing speech/non-speech state transitions.

    Implements:
    - Onset confirmation hysteresis (requires N consecutive candidate frames to declare SPEECH).
    - Hangover smoothing (holds SPEECH/HANGOVER state for M frames after speech drops below threshold).
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        """Initialize the state machine with configuration.

        Args:
            config: VADConfig specifying hangover_frames and onset_frames.
        """
        self.config = config or VADConfig()
        self._state: VADState = VADState.SILENCE
        self._onset_counter: int = 0
        self._hangover_counter: int = 0

    @property
    def state(self) -> VADState:
        """Current state of the state machine."""
        return self._state

    @property
    def primary_state(self) -> VADState:
        """Coarse binary state: SPEECH if in speech or hangover, else SILENCE."""
        return VADState.SPEECH if self.is_speech else VADState.SILENCE

    @property
    def is_speech(self) -> bool:
        """Return True if the current state corresponds to speech (including hangover)."""
        return self._state in (VADState.SPEECH, VADState.HANGOVER)

    @property
    def onset_counter(self) -> int:
        """Number of consecutive speech candidate frames currently accumulated."""
        return self._onset_counter

    @property
    def hangover_counter(self) -> int:
        """Number of hangover frames remaining before returning to SILENCE."""
        return self._hangover_counter

    def step(self, is_speech_candidate: bool) -> VADState:
        """Update state machine based on the current frame's speech candidate flag.

        Args:
            is_speech_candidate: Boolean indicator from feature scoring/thresholding.

        Returns:
            The updated VADState after applying state transition logic.
        """
        if is_speech_candidate:
            if self._state == VADState.SILENCE:
                self._onset_counter += 1
                if self._onset_counter >= self.config.onset_frames:
                    self._state = VADState.SPEECH
                    self._hangover_counter = self.config.hangover_frames
                else:
                    self._state = VADState.POSSIBLE_ONSET
            elif self._state == VADState.POSSIBLE_ONSET:
                self._onset_counter += 1
                if self._onset_counter >= self.config.onset_frames:
                    self._state = VADState.SPEECH
                    self._hangover_counter = self.config.hangover_frames
            else:
                # Already in SPEECH or HANGOVER
                self._state = VADState.SPEECH
                self._hangover_counter = self.config.hangover_frames
        else:
            self._onset_counter = 0
            if self._state in (VADState.SPEECH, VADState.HANGOVER):
                if self._hangover_counter > 0:
                    self._hangover_counter -= 1
                    self._state = VADState.HANGOVER
                else:
                    self._state = VADState.SILENCE
            else:
                self._state = VADState.SILENCE

        return self._state

    def reset(self) -> None:
        """Reset state machine to initial silence state."""
        self._state = VADState.SILENCE
        self._onset_counter = 0
        self._hangover_counter = 0
