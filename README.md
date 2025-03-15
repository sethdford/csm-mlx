# csm-mlx

An implementation of the CSM(Conversation Speech Model) for Apple Silicon using MLX.

## Installation

Recommendation: Give [`uv`](https://docs.astral.sh/uv/) a try. It's truly magical.
```bash
uv add git+https://github.com/senstella/csm-mlx
```

Or, you can install it via pip:
```bash
pip install git+https://github.com/senstella/csm-mlx
```

## Usage

### Basic generation
```python
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import hf_hub_download
from csm_mlx import CSM, csm_1b, generate

# Initialize the model
csm = CSM(csm_1b())  # csm_1b() is a configuration for the CSM model.
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

# Generate audio from text
audio = generate(
    csm,
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
    sampler=make_sampler(temp=0.5, min_p=0.1), # Put mlx_lm's sampler here! Supports: temp, top_p, min_p, min_tokens_to_keep, top_k.
    # Additionally, you can provide `stream` argument to specify what device to use for generation.
    # https://ml-explore.github.io/mlx/build/html/usage/using_streams.html
)

# Save the generated audio; Install numpy, torchaudio, audiofile to run!
import numpy as np
import torch
import torchaudio

torchaudio.save("audio.wav", torch.Tensor(np.asarray(audio)).unsqueeze(0).cpu(), 24_000)
```

### Providing Context
```python
from csm_mlx import CSM, csm_1b, generate, Segment
import mlx.core as mx

# Initialize the model
csm = CSM(csm_1b())
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

# Create previous conversation segments
context = [
    Segment(
        speaker=0,
        text="How are you doing today?",
        audio=mx.array(...)  # Previous audio for this segment
    ),
    Segment(
        speaker=1,
        text="I'm doing great, thank you!",
        audio=mx.array(...)  # Previous audio for this segment
    )
]

# Generate a response in the conversation
audio = generate(
    csm,
    text="That's wonderful to hear!",
    speaker=0,
    context=context,
    max_audio_length_ms=5_000
    # If you don't provide any sampler, greedy sampling will be used.
)
```

## Todo

- [X] Fix up RoPE
- [ ] Implement watermarking
- [ ] Add streaming generation
- [ ] Optimize performance further for real-time inference

## Acknowledgments

- Thanks to [Sesame](https://sesame.com) for [original PyTorch implementation](https://github.com/SesameAILabs/csm) and [weights](https://huggingface.co/sesame/csm-1b)!
- Thanks to [torchtune](https://github.com/pytorch/torchtune) project for providing LLaMA attention implementation.
- Thanks to [MLX](https://github.com/ml-explore/mlx) project for providing the framework that made this implementation possible.

## License

Apache 2.0
