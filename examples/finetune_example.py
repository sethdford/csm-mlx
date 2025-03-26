#!/usr/bin/env python3
"""Example script for finetuning CSM on a custom dataset and generating audio with the result."""

import os
import argparse
import audiofile
import numpy as np

import mlx.core as mx
import mlx.optimizers as optim
from huggingface_hub import hf_hub_download

from csm_mlx import CSM, csm_1b, generate, Segment
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Example finetuning and generation for CSM")
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./finetune_output",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, this is a test of my finetuned voice.",
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
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate for finetuning"
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
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = CSMDataset.from_json(args.data_path)
    print(f"Dataset contains {len(dataset)} samples")
    
    # Initialize optimizer
    print(f"Initializing optimizer with learning rate {args.learning_rate}...")
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = CSMTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.output_dir,
        save_every=50,
        log_every=10,
    )
    
    # Start finetuning
    print(f"Starting finetuning for {args.epochs} epochs...")
    training_history = trainer.train(
        dataset=dataset,
        batch_size=4,
        epochs=args.epochs,
    )
    
    print("Finetuning complete!")
    
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
    output_audio_path = os.path.join(args.output_dir, "generated_audio.wav")
    audiofile.write(output_audio_path, np.asarray(audio), 24000)
    print(f"Generated audio saved to {output_audio_path}")


if __name__ == "__main__":
    main() 