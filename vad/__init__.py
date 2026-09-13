"""Voice Activity Detector (VAD) package.

A lightweight, local, explainable Voice Activity Detector built from scratch
without external APIs, cloud services, or pretrained ML models.
"""

from vad.config import VADConfig
from vad.audio import AudioFrame, load_wav, frame_audio
from vad.features import FrameFeatures, FeatureExtractor
from vad.state_machine import VADState, VADStateMachine
from vad.detector import VoiceActivityDetector, SpeechSegment, FrameDecision

__all__ = [
    "VADConfig",
    "AudioFrame",
    "load_wav",
    "frame_audio",
    "FrameFeatures",
    "FeatureExtractor",
    "VADState",
    "VADStateMachine",
    "VoiceActivityDetector",
    "SpeechSegment",
    "FrameDecision",
]

__version__ = "0.1.0"
