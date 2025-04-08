from pathlib import Path

import audiofile
import audresample
import mlx.core as mx
import numpy as np


def read_audio(filename: Path, sampling_rate: int) -> mx.array:
    signal, original_sampling_rate = audiofile.read(str(filename), always_2d=True)

    signal = audresample.resample(signal, original_sampling_rate, sampling_rate)

    signal = mx.array(signal)

    if signal.shape[0] >= 1:
        signal = signal.mean(axis=0)
    else:
        signal = signal.squeeze(0)

    return signal  # (audio_length, )


def write_audio(array: mx.array, filename: Path, sampling_rate: int):
    audiofile.write(
        str(filename), np.asarray(array), sampling_rate
    )  # audiofile automatically recognizes dimension the array has!
