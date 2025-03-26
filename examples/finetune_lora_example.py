#!/usr/bin/env python3
"""Example script for LoRA finetuning of CSM model on a custom dataset."""

import os
import argparse
import audiofile
import numpy as np

import mlx.core as mx
import mlx.optimizers as optim
from huggingface_hub import hf_hub_download

from csm_mlx import CSM, csm_1b, generate, Segment
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.lora import apply_lora_to_model, load_lora_weights, merge_lora_weights
from csm_mlx.finetune.finetune_lora import LoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Example LoRA finetuning for CSM")
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./lora_finetune_output",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, this is a test of my LoRA finetuned voice.",
        help="Text to generate after finetuning"
    )
    parser.add_argument(
        "--speaker-id", type=int, default=0,
        help="Speaker ID to use for generation"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of epochs to finetune"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4,
        help="Learning rate for finetuning"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="Rank for LoRA adapters"
    )
    parser.add_argument(
        "--merge-weights", action="store_true",
        help="Merge LoRA weights into base model after training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Creating output directory...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the model
    print("Initializing model...")
    model = CSM(csm_1b())
    
    # Load pretrained weights
    print("Loading pretrained weights...")
    weight_path = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
    model.load_weights(weight_path)
    
    # Apply LoRA to the model
    print(f"Applying LoRA with rank={args.lora_rank}")
    target_modules = ["attn", "codebook0_head", "projection"]
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,  # Common practice to set alpha = 2 * rank
        target_modules=target_modules
    )
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = CSMDataset.from_json(args.data_path)
    print(f"Dataset contains {len(dataset)} samples")
    
    # Initialize optimizer
    print(f"Initializing optimizer with learning rate {args.learning_rate}...")
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Initialize trainer
    print("Initializing LoRA trainer...")
    trainer = LoRATrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.output_dir,
        save_every=50,
        log_every=10,
    )
    
    # Start finetuning
    print(f"Starting LoRA finetuning for {args.epochs} epochs...")
    training_history = trainer.train(
        dataset=dataset,
        batch_size=4,
        epochs=args.epochs,
    )
    
    print("Finetuning complete!")
    
    # Get final checkpoint path
    final_checkpoint = os.path.join(args.output_dir, f"lora_ckpt_epoch_{args.epochs}.safetensors")
    
    # Optionally merge weights
    if args.merge_weights:
        print("Merging LoRA weights into base model...")
        model = merge_lora_weights(model)
        print("Weights merged - LoRA adapters removed")
    
    # Generate audio with the finetuned model
    print(f"Generating audio for prompt: '{args.prompt}'")
    audio = generate(
        model,
        text=args.prompt,
        speaker=args.speaker_id,
        context=[],
        max_audio_length_ms=10_000,
    )
    
    # Save the generated audio
    output_audio_path = os.path.join(args.output_dir, "lora_generated_audio.wav")
    audiofile.write(output_audio_path, np.asarray(audio), 24000)
    print(f"Generated audio saved to {output_audio_path}")
    
    # Save example usage
    usage_example = f"""
# Example code for using this LoRA checkpoint:

from csm_mlx import CSM, csm_1b, generate
from finetune.lora import apply_lora_to_model, load_lora_weights

# Initialize model
model = CSM(csm_1b())
model.load_weights("{weight_path}")

# Apply LoRA structure (with same parameters as training)
model = apply_lora_to_model(
    model,
    rank={args.lora_rank},
    alpha={args.lora_rank * 2},
    target_modules={target_modules}
)

# Load LoRA weights
model = load_lora_weights(model, "{final_checkpoint}")

# Generate audio
audio = generate(
    model,
    text="Your text here",
    speaker={args.speaker_id},
    context=[],
    max_audio_length_ms=10000
)
    """
    
    with open(os.path.join(args.output_dir, "usage_example.py"), "w") as f:
        f.write(usage_example)
    
    print(f"Usage example saved to {os.path.join(args.output_dir, 'usage_example.py')}")


if __name__ == "__main__":
    main() 