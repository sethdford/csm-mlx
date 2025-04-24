from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx

from csm_mlx.utils import read_audio

SAMPLING_RATE = 24000


@dataclass
class Segment:
    speaker: int
    text: str
    _audio: Optional[mx.array] = None
    audio_path: Optional[Path] = None

    def __post_init__(self):
        if self._audio is None and self.audio_path is None:
            raise ValueError("Either 'audio' or 'audio_path' must be provided")

    @property
    def audio(self):
        if self._audio is not None:
            return self._audio
        elif self.audio_path is not None:
            return read_audio(self.audio_path, SAMPLING_RATE)

        raise ValueError("Neither 'audio' nor 'audio_path' is provided")

    @audio.setter
    def audio(self, value):
        self._audio = value

    def __init__(
        self,
        speaker: int,
        text: str,
        audio: Optional[mx.array] = None,
        audio_path: Optional[Path] = None,
    ):
        self.speaker = speaker
        self.text = text
        self._audio = audio
        self.audio_path = audio_path
