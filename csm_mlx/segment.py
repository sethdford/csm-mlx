from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: mx.array
