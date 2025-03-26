# Sample Dataset for CSM-MLX Finetuning

This directory contains a sample dataset structure for finetuning CSM-MLX models.

## Dataset Structure

The sample dataset consists of:
- `sample_dataset.json`: JSON file containing text-audio pairs
- `audio/`: Directory where audio files should be placed

## How to Use This Dataset

1. **Replace with Real Audio**: 
   - Replace the placeholder files in the `audio/` directory with real audio recordings
   - Each audio file should be a WAV file sampled at 24kHz (mono)
   - The recordings should be clear and of good quality

2. **Update the JSON**:
   - Make sure the text accurately transcribes the audio content
   - Update paths if necessary to match your actual audio files
   - Assign appropriate speaker IDs (0, 1, 2, etc.) for different speakers

## Creating Your Own Dataset

### Recording Audio

For best results:
- Use a good quality microphone
- Record in a quiet environment with minimal background noise
- Speak clearly and at a natural pace
- Keep a consistent distance from the microphone
- Aim for 5-10 minutes of audio per speaker for basic adaptation, more for better results

### Processing Audio

All audio files should be:
- WAV format
- 24kHz sampling rate
- Mono channel
- Properly trimmed (minimal silence at start/end)

You can use tools like Audacity or FFmpeg to convert and process your audio:

```bash
# Example FFmpeg command to convert audio to the right format
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
```

### Using the Dataset Preparation Script

Use the included script to prepare your dataset from audio files and a transcript:

```bash
python -m finetune.prepare_dataset \
  --audio-dir /path/to/your/audio/files \
  --text-file /path/to/your/transcript.txt \
  --output-json finetune/datasets/your_dataset.json
```

## Sample Transcript Format

For reference, here's how a transcript file should look:

```
sample_001.wav|Hello, this is a sample voice for finetuning.
sample_002.wav|The quick brown fox jumps over the lazy dog.
```

Or with speaker IDs:

```
sample_001.wav|0|Hello, this is a sample voice for finetuning.
sample_002.wav|0|The quick brown fox jumps over the lazy dog.
sample_006.wav|1|This is an example of another speaker's voice.
```

## Finetuning with this Dataset

```bash
# For regular finetuning
python -m finetune.finetune --data-path finetune/datasets/sample_dataset.json

# For LoRA finetuning (recommended)
python -m finetune.finetune_lora --data-path finetune/datasets/sample_dataset.json
``` 