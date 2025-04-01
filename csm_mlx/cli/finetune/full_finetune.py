import os
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.optimizers as optim
import typer
from huggingface_hub import hf_hub_download
from typing_extensions import Annotated

from csm_mlx import CSM, csm_1b
from csm_mlx.cli.config import Models, OptimizerChoice
from csm_mlx.finetune.dataset import CSMDataset
from csm_mlx.finetune.trainer import CSMTrainer

app = typer.Typer(no_args_is_help=True)


@app.command("full")
def finetune_command(
    data_path: Annotated[
        Path,
        typer.Option(
            "--data-path",
            help="Path to JSON dataset file with text/audio pairs",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        Models,
        typer.Option(
            "--model",
            "-m",
            help="Model size to use (currently only 1b is supported)",
        ),
    ] = Models._1b,
    pretrained_path: Annotated[
        Optional[Path],
        typer.Option(
            "--pretrained-path",
            help="Path to pretrained model weights. If not provided, will download from HF.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    resume_from: Annotated[
        Optional[Path],
        typer.Option(
            "--resume-from",
            help="Path to checkpoint to resume training from",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    max_samples: Annotated[
        Optional[int],
        typer.Option(
            "--max-samples",
            help="Maximum number of samples to use from dataset",
            min=1,
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for training",
            min=1,
        ),
    ] = 4,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of epochs to train for",
            min=1,
        ),
    ] = 5,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            "--lr",
            help="Learning rate for optimizer",
            min=0.0,
        ),
    ] = 1e-5,
    weight_decay: Annotated[
        float,
        typer.Option(
            "--weight-decay",
            "--wd",
            help="Weight decay for optimizer",
            min=0.0,
        ),
    ] = 1e-4,
    save_every: Annotated[
        int,
        typer.Option(
            "--save-every",
            help="Save checkpoints every N steps",
            min=1,
        ),
    ] = 100,
    log_every: Annotated[
        int,
        typer.Option(
            "--log-every",
            help="Log metrics every N steps",
            min=1,
        ),
    ] = 10,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save checkpoints and logs",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("./finetune_output"),
    freeze_backbone: Annotated[
        bool,
        typer.Option(
            "--freeze-backbone/--no-freeze-backbone",
            help="Freeze backbone parameters during finetuning",
        ),
    ] = False,
    freeze_decoder: Annotated[
        bool,
        typer.Option(
            "--freeze-decoder/--no-freeze-decoder",
            help="Freeze decoder parameters during finetuning",
        ),
    ] = False,
    optimizer: Annotated[
        OptimizerChoice,
        typer.Option(
            "--optimizer",
            help="Optimizer to use for training",
            case_sensitive=False,
        ),
    ] = OptimizerChoice.ADAMW,
):
    """Full finetuning for CSM models."""
    print("Starting finetuning script...")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing model...")
    if model == Models._1b:
        model_args = csm_1b()
    else:
        raise typer.BadParameter(
            f"Unsupported model size: {model.value}. Only '1b' is currently supported."
        )

    csm_model = CSM(model_args)
    mx.eval(csm_model.parameters())

    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        csm_model.load_weights(str(pretrained_path))
    elif resume_from:
        print("Weights will be loaded from resume checkpoint.")
        pass
    else:
        print("Downloading pretrained weights from Hugging Face...")
        if model == Models._1b:
            repo_id = "senstella/csm-1b-mlx"
            filename = "ckpt.safetensors"
        else:
            raise ValueError("Cannot determine Hugging Face repo for unknown model.")

        print(f"Downloading {filename} from {repo_id}...")
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"Downloaded weights to {weight_path}")
        csm_model.load_weights(weight_path)

    if freeze_backbone:
        print("Freezing backbone parameters...")
        csm_model.backbone.freeze()

    if freeze_decoder:
        print("Freezing decoder parameters...")
        csm_model.decoder.freeze()

    mx.eval(csm_model.parameters())

    csm_model.train()

    print(
        f"Initializing {optimizer.value} optimizer with lr={learning_rate}, weight_decay={weight_decay}"
    )
    if optimizer == OptimizerChoice.ADAM:
        if weight_decay > 0:
            print(
                f"Warning: Weight decay {weight_decay} requested for Adam optimizer, but MLX Adam does not support it. Ignoring weight decay."
            )
        opt = optim.Adam(learning_rate=learning_rate)
    elif optimizer == OptimizerChoice.SGD:
        opt = optim.SGD(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == OptimizerChoice.ADAMW:
        opt = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    trainer = CSMTrainer(
        model=csm_model,
        optimizer=opt,
        checkpoint_dir=str(output_dir),
        save_every=save_every,
        log_every=log_every,
    )

    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        trainer.load_checkpoint(str(resume_from))

    print(f"Loading dataset from {data_path}")
    dataset = CSMDataset.from_json(
        str(data_path),
        max_samples=max_samples,
    )
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check the data path and format.")
        raise typer.Exit(code=1)
    if len(dataset) < batch_size:
        print(
            f"Warning: Dataset size ({len(dataset)}) is smaller than batch size ({batch_size}). Consider reducing batch size."
        )

    print(f"Starting training for {epochs} epochs, batch size {batch_size}")
    print(f"Optimizer: {optimizer.value}, LR: {learning_rate}, WD: {weight_decay}")
    print(f"Saving checkpoints every {save_every} steps to {output_dir}")
    print(f"Logging metrics every {log_every} steps")
    print(f"Backbone frozen: {freeze_backbone}, Decoder frozen: {freeze_decoder}")

    try:
        _training_history = trainer.train(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
        )
        print("\nTraining complete!")
        print(f"Checkpoints and logs saved in {output_dir}")
        final_save_path = output_dir / "final_model.safetensors"
        print(f"Saving final model weights to {final_save_path}...")
        csm_model.save_weights(str(final_save_path))
        print("Final model saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
