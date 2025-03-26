"""Finetuning module for CSM-MLX models."""

# Path handling isn't needed anymore since we're inside the csm_mlx package
from csm_mlx.finetune.dataset import CSMDataset, AudioTextSample
from csm_mlx.finetune.trainer import CSMTrainer
from csm_mlx.finetune.lora import (
    LoRALinear, 
    apply_lora_to_model, 
    load_lora_weights, 
    merge_lora_weights,
    trainable_params
)
from csm_mlx.finetune.finetune_lora import LoRATrainer 