"""Finetuning module for CSM-MLX models."""

# Path handling isn't needed anymore since we're inside the csm_mlx package
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer, TrainArgs
from csm_mlx.finetune.utils import load_adapters

__all__ = [
    "CSMTrainer",
    "CSMDataset",
    "TrainArgs",
    "load_adapters",
]
