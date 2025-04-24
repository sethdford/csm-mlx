# CLI Reference for fine-tuning!

## Full Finetuning

```
csm-mlx finetune full [OPTIONS]
```

### Options

- `--data-path PATH`: Path to the JSON dataset [required] (Look [data preparation](#data-preparation) for dataset format and helper command!)
- `--output-dir PATH`: Directory to save outputs [required]
- `--model [1b]`: Model size (default: 1b)
- `--pretrained-path PATH`: Path to local weights
- `--epochs INT`: Number of training epochs (default: 5)
- `--batch-size INT`: Batch size for training (default: 4)
- `--learning-rate, --lr FLOAT`: Learning rate (default: 1e-5)
- `--weight-decay, --wd FLOAT`: Weight decay for regularization (default: 1e-4)
- `--freeze-backbone`: Freeze encoder backbone
- `--freeze-decoder`: Freeze decoder layers
- `--max-audio-length-ms INT`: Maximum audio length in milliseconds
- `--mask-speaker-ids LIST`: Comma-separated speaker IDs to ignore in loss
  - Pretty similar to masking user's turn when training LLM. Makes it won't train on the voice you don't want.
- `--max-norm FLOAT`: Gradient clipping norm (0.0 disables clipping)
- `--first_codebook_weight_multiplier FLOAT`: Weight multiplier for first codebook
  - First codebook computed by backbone, rest computed by decoder!
- `--ckpt_freq INT`: Save checkpoint every N steps
- `--log-freq INT`: Log metrics every N steps
- `--gradient-ckpt`: Enable gradient checkpointing
- `--optimizer [adamw|adam|sgd]`: Optimizer selection

### Outputs

- Final model: `final_model.safetensors`
- Training logs and checkpoints with optimizer state dumps

### Example

```bash
csm-mlx finetune full \
    --data-path training_data.json \
    --output-dir ./finetune_output_full \
    --model 1b \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-5
```

## LoRA Finetuning

```
csm-mlx finetune lora [OPTIONS]
```

### Options

- `--data-path PATH`: Path to the JSON dataset [required] (Look [data preparation](#data-preparation) for dataset format and helper command!)
- `--output-dir PATH`: Directory to save outputs [required]
- `--model [1b]`: Model size (default: 1b)
- `--pretrained-path PATH`: Path to local weights
- `--epochs INT`: Number of training epochs (default: 10)
- `--batch-size INT`: Batch size for training (default: 8)
- `--learning-rate, --lr FLOAT`: Learning rate (default: 5e-4)
- `--weight-decay, --wd FLOAT`: Weight decay (default: 1e-4)
- `--lora-rank INT`: Rank of LoRA matrices (default: 8)
- `--lora-alpha INT`: LoRA scaling factor (default: 16)
- `--target-modules LIST`: Layer types for LoRA (e.g., "attn projection")
  - `attn` includes: all QKVO projection layers, all MLP layers. Use mindfully!
- `--train-embeddings`: Train embedding layers directly
- `--only-save-adapter`: Save only adapter weights (default: True) - will still save optimizer states!
- `--max-audio-length-ms INT`: Maximum audio length in milliseconds
- `--mask-speaker-ids LIST`: Comma-separated speaker IDs to ignore in loss
  - Pretty similar to masking user's turn when training LLM. Makes it won't train on the voice you don't want.
- `--max-norm FLOAT`: Gradient clipping norm
- `--first_codebook_weight_multiplier FLOAT`: Weight multiplier for first codebook
  - First codebook computed by backbone, rest computed by decoder!
- `--ckpt_freq INT`: Save checkpoint every N steps
- `--log-freq INT`: Log metrics every N steps
- `--gradient-ckpt`: Enable gradient checkpointing
- `--optimizer [adamw|adam|sgd]`: Optimizer selection

### Outputs

- Adapter weights: `adapters.safetensors`
- Configuration: `adapter_config.json`
- Training logs and checkpoints with optimizer state dumps

### Example

```bash
csm-mlx finetune lora \
    --data-path training_data.json \
    --output-dir ./finetune_output_lora \
    --model 1b \
    --lora-rank 8 \
    --lora-alpha 16 \
    --target-modules attn projection
```

## Using Finetuned Models

### Full Finetuned Models
Load `final_model.safetensors` directly into your CSM model for inference.

### LoRA Models
1. Load the original pre-trained model
2. Apply the adapter weights (`adapters.safetensors`) using `adapter_config.json`

## Data Preparation

> Might want to use with `mask_speaker_ids`!
```json
[
  [
    {
      "text": "Hello!",
      "audio_path": "~/wherever/audio.wav",
      "speaker_id": 0
    },
    {
      "text": "Great to see you!",
      "audio_path": "~/wherever/audio454.wav",
      "speaker_id": 9
    },
  ],
  [
    {
      "text": "Good day!",
      "audio_path": "./audio234.mp3",
      "speaker_id": 9
    },
    {
      "text": "Great to see you!",
      "audio_path": "../audio12343.wav",
      "speaker_id": 0
    },
  ]
]
```

### Helper command for converting folder style dataset

```
csm-mlx finetune convert <INPUT_DIR> <OUTPUT_JSON>
```

#### Arguments

- `INPUT_DIR`: Directory containing conversation subdirectories
- `OUTPUT_JSON`: Path for output JSON file

> The script sorts the audio files within each conversation directory using a natural_sort_key. This means it tries to sort filenames containing numbers intelligently (e.g., 1, 2, 10 instead of 1, 10, 2). To ensure your utterances are processed in the correct conversational order, it's highly recommended to prefix your filenames with a sequence number. Also, all filenames must include `speaker{n}_` such in way you can see in the example for correct speaker identification!

#### Directory Structure

```
my_raw_dataset/                <-- input_dir
├── conversation_01/
│   ├── 001_speaker0_hello_there.wav      <-- Speaker ID 0, sequence 1
│   ├── 001_speaker0_hello_there.txt      <-- Matches base name
│   ├── 002_speaker1_general_kenobi.mp3   <-- Speaker ID 1, sequence 2
│   ├── 002_speaker1_general_kenobi.txt   <-- Matches base name
│   ├── 003_speaker0_you_are_bold.flac    <-- Speaker ID 0, sequence 3
│   └── 003_speaker0_you_are_bold.txt    <-- Matches base name
│
├── meeting_minutes/
│   ├── 01_speaker5_intro.wav      <-- Speaker ID 5, sequence 1
│   ├── 01_speaker5_intro.txt      <-- Matches base name
│   ├── 02_speaker12_agenda.aac    <-- Speaker ID 12, sequence 2
│   ├── 02_speaker12_agenda.txt    <-- Matches base name
│   ├── 03_speaker5_point_one.m4a  <-- Speaker ID 5, sequence 3
│   ├── 03_speaker5_point_one.txt  <-- Matches base name
│   ├── 04_speaker12_comment.ogg   <-- Speaker ID 12, sequence 4
│   └── 04_speaker12_comment.txt   <-- Matches base name
│
└── other_audio.wav             <-- This file will be ignored (not in a subdir)
```

#### Example

```bash
csm-mlx finetune convert ./my_audio_dataset ./training_data.json
```
