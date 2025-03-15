# csm-mlx

Implementation of the CSM(Conversation Speech Model) for Apple Silicon using MLX.

## Installation

Recommendation: Use [`uv`](https://docs.astral.sh/uv/) to install it. It's truly magical.
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
csm = CSM(csm_1b())
weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
csm.load_weights(weight)

# Generate audio from text
audio = generate(
    csm,
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
    sampler=make_sampler(temp=0.5, min_p=0.1),
)

# Save the generated audio; Of course, you have to install those dependencies. Or maybe you can use a different library.
import numpy as np
import torch
import torchaudio

torchaudio.save("audio.wav", torch.Tensor(np.asarray(audio)).unsqueeze(0).cpu(), 24_000)
```

### Provide the context
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

# Save or do anything you want.
```

`csm_1b()` is a configuration for the CSM model.

## Todo

- [X] Fix up RoPE
- [ ] Implement watermarking
- [ ] Add streaming generation
- [ ] Optimize performance futher for real-time inference

## Acknowledgments

Thank you for [Sesame](https://sesame.com) for original PyTorch implementation and weights!
