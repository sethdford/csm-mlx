import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import cross_entropy
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import grad_checkpoint
from tqdm import tqdm

from csm_mlx.finetune.dataset import CSMDataset, CSMPairwiseDataset
from csm_mlx.models import CSM


@dataclass
class TrainArgs:
    model: CSM
    optimizer: optim.Optimizer
    output_dir: Path
    first_codebook_weight_multiplier: float = 1.0
    max_norm: float = 1.0
    gradient_checkpointing: bool = False
    log_freq: int = 1
    ckpt_freq: int = 1
    only_save_trainable_params: bool = False


@dataclass
class DPOArgs(TrainArgs):
    beta: float = 0.1


@dataclass
class KTOArgs(TrainArgs):
    reference_model: CSM | None = None
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    learning_rate: float = 0.0


@dataclass
class TrainingRecord:
    step: int
    epoch: int
    loss: float
    learning_rate: float


class History:
    def __init__(self):
        self.records: List[TrainingRecord] = []

    def log(self, step: int, epoch: int, loss: float, lr: float):
        self.records.append(TrainingRecord(step, epoch, loss, lr))

    @property
    def state(self):
        return [asdict(record) for record in self.records]

    @state.setter
    def state(self, records: List[Dict]):
        self.records = [TrainingRecord(**record) for record in records]


class CheckpointManager:
    def __init__(
        self,
        model: CSM,
        optimizer: optim.Optimizer,
        state: TrainerState,
        history: History,
        checkpoint_dir: Path,
        only_save_trainable_params: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.history = history
        self.dir = checkpoint_dir
        self.only_save_trainable_params = only_save_trainable_params
        os.makedirs(self.dir, exist_ok=True)

    def save(self):
        suffix = f"step_{self.state.step}"
        weights_name = "latest.safetensors"
        optimizer_state_name = "optimizer_state.safetensors"
        state_name = "trainer_state.json"

        # Trainer state
        trainer_state = {
            "trainer_state": asdict(self.state),
            "history": self.history.state,
        }

        # Local
        os.makedirs(os.path.join(self.dir, suffix), exist_ok=True)
        mx.save_safetensors(
            os.path.join(self.dir, suffix, weights_name),
            dict(
                tree_flatten(
                    self.model.parameters()
                    if not self.only_save_trainable_params
                    else self.model.trainable_parameters()
                )
            ),
        )
        with open(os.path.join(self.dir, suffix, state_name), "w") as f:
            json.dump(trainer_state, f, indent=2)
        mx.save_safetensors(
            os.path.join(self.dir, suffix, optimizer_state_name),
            dict(tree_flatten(self.optimizer.state)),
        )

        # Global
        mx.save_safetensors(
            os.path.join(self.dir, weights_name),
            dict(
                tree_flatten(
                    self.model.parameters()
                    if not self.only_save_trainable_params
                    else self.model.trainable_parameters()
                )
            ),
        )
        mx.save_safetensors(
            os.path.join(self.dir, optimizer_state_name),
            dict(tree_flatten(self.optimizer.state)),
        )

        with open(os.path.join(self.dir, state_name), "w") as f:
            json.dump(trainer_state, f, indent=2)

        print(f"Saved checkpoint (step {self.state.step})")

    def load(self):
        weights_path = os.path.join(self.dir, "latest.safetensors")
        state_path = os.path.join(self.dir, "trainer_state.json")
        optimizer_state_path = os.path.join(self.dir, "optimizer_state.safetensors")

        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded latest run weights from {weights_path}")

        if os.path.exists(optimizer_state_path):
            self.optimizer.state = mx.load(os.path.join(self.dir, optimizer_state_path))
            print(f"Loaded optimizer state from {optimizer_state_path}")

        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                trainer_state = json.load(f)

            ts = trainer_state["trainer_state"]
            self.state.step = ts["step"]
            self.state.epoch = ts["epoch"]
            self.state.learning_rate = ts["learning_rate"]

            self.history.state = trainer_state["history"]

            print(f"Loaded trainer state (step {self.state.step})")
        else:
            print("Trainer state not found. Starting fresh training.")


class CSMTrainer:
    """CSM SFT Trainer."""

    def __init__(self, args: TrainArgs):
        self.model = args.model
        self.optimizer = args.optimizer
        self.args = args

        self.state = TrainerState(learning_rate=float(self.optimizer.learning_rate))
        self.history = History()
        self.checkpointer = CheckpointManager(
            self.model,
            self.optimizer,
            self.state,
            self.history,
            args.output_dir,
        )
        self.checkpointer.load()

        if self.args.gradient_checkpointing:
            grad_checkpoint(self.model.backbone.layers[0])
            grad_checkpoint(self.model.decoder.layers[0])

        # Compile the loss and train functions
        self._step_fn = None

    @staticmethod
    def compute_loss(
        model: CSM,
        batch: Dict[str, mx.array],
        *,
        per_sample: bool = False,
        cause_mismatch: bool = False,
        **kwargs,
    ) -> mx.array:
        """Compute loss for a batch of samples."""
        tokens = batch["tokens"]
        masks = batch["masks"]
        loss_masks = batch["loss_masks"]
        first_codebook_weight_multiplier = batch["first_codebook_weight_multiplier"]

        batch_size, seq_len, n_codebooks = tokens.shape

        # Extract text tokens (last codebook) and audio tokens (all other codebooks)
        audio_tokens = tokens[:, :, :-1]  # (batch, seq, codebook)
        shifted_audio_tokens = audio_tokens[:, 1:, :]  # (batch, seq - 1, codebook)

        audio_masks = masks[:, :, :-1]  # (batch, seq, codebook)
        shifted_audio_masks = audio_masks[:, 1:, :]  # (batch, seq - 1, codebook)

        audio_loss_masks = loss_masks[:, :, :-1]  # (batch, seq, codebook)
        shifted_audio_loss_masks = audio_loss_masks[
            :, 1:, :
        ]  # (batch, seq - 1, codebook)

        # Forward pass through the model
        backbone_embeds = model.embed_tokens(tokens)
        backbone_embeds = backbone_embeds * mx.expand_dims(masks, axis=-1)
        backbone_input = backbone_embeds.sum(-2)
        backbone_input = backbone_input[
            :, :-1, :
        ]  # (batch, seq - 1, embed_dim) - we don't need next token prediction here

        backbone_hidden = model.backbone(backbone_input)  # (batch, seq - 1, embed_dim)

        c0_logits = model.codebook0_head(backbone_hidden)

        ci_stacked = mx.stack(
            [
                model.embed_audio(i, shifted_audio_tokens[:, :, i])
                for i in range(model.n_audio_codebooks)
            ],
            axis=-2,
        )  # (batch, seq - 1, codebook, embed_dim)
        decoder_inputs = mx.concat(
            [mx.expand_dims(backbone_hidden, axis=-2), ci_stacked], axis=-2
        )  # (batch, seq - 1, codebook + 1(backbone activation), embed_dim)

        decoder_inputs = decoder_inputs.reshape(
            -1, n_codebooks, decoder_inputs.shape[-1]
        )
        # TODO: Apply compute amortization since those consumes VERY HIGH memory as mentioned in Sesame's blog
        # https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
        decoder_hidden = model.decoder(model.projection(decoder_inputs))
        decoder_hidden = decoder_hidden.reshape(
            batch_size, seq_len - 1, n_codebooks, -1
        )[
            :, :, 1:-1, :
        ]  # (batch, seq - 1, codebook - 1, vocab_size) - we don't need c0 predictions and c_last predictions (doesn't exists)

        # Calculate total losses at once.
        shifted_audio_loss_masks = mx.logical_and(
            shifted_audio_masks, shifted_audio_loss_masks
        )

        if cause_mismatch:
            shifted_audio_tokens = mx.concat(
                [shifted_audio_tokens[:, 1:], shifted_audio_tokens[:, :1]], axis=1
            )

        c0_loss = cross_entropy(
            c0_logits.reshape(-1, c0_logits.shape[-1]),
            shifted_audio_tokens[:, :, 0].reshape(-1),
            reduction="none",
        )
        if per_sample:
            c0_loss = (
                (
                    c0_loss.reshape(batch_size, -1) * shifted_audio_loss_masks[:, :, 0]
                ).sum(-1)
                / shifted_audio_loss_masks[:, :, 0].sum(-1)
                * first_codebook_weight_multiplier
            )
        else:
            c0_loss = (
                (c0_loss * shifted_audio_loss_masks[:, :, 0].reshape(-1)).sum()
                / shifted_audio_loss_masks[:, :, 0].sum()
                * first_codebook_weight_multiplier
            )

        total_loss = c0_loss / model.n_audio_codebooks

        for index in range(1, model.n_audio_codebooks):
            ci_logits = mx.matmul(
                decoder_hidden[:, :, index - 1, :], model.audio_head[index - 1]
            )
            ci_loss = cross_entropy(
                ci_logits.reshape(-1, ci_logits.shape[-1]),
                shifted_audio_tokens[:, :, index].reshape(-1),
                reduction="none",
            )
            if per_sample:
                ci_loss = (
                    ci_loss.reshape(batch_size, -1)
                    * shifted_audio_loss_masks[:, :, index]
                ).sum(-1) / shifted_audio_loss_masks[:, :, index].sum(-1)
            else:
                ci_loss = (
                    ci_loss * shifted_audio_loss_masks[:, :, index].reshape(-1)
                ).sum() / shifted_audio_loss_masks[:, :, index].sum()
            total_loss += ci_loss / model.n_audio_codebooks

        return total_loss

    def train_step(self, batch: Dict[str, mx.array]) -> float:
        """Perform a single training step."""

        model = self.model
        optimizer = self.optimizer
        first_codebook_weight_multiplier = mx.array(
            self.args.first_codebook_weight_multiplier
        )

        state = [
            model.state,
            optimizer.state,
            mx.random.state,
        ]

        if self._step_fn is None:
            loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)  # type: ignore

            if self.args.max_norm > 0:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "first_codebook_weight_multiplier": first_codebook_weight_multiplier,
                        },
                    )

                    grads, norm = optim.clip_grad_norm(grads, self.args.max_norm)

                    optimizer.update(model, grads)

                    return loss, norm

                self._step_fn = _step
            else:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "first_codebook_weight_multiplier": first_codebook_weight_multiplier,
                        },
                    )

                    optimizer.update(model, grads)

                    return loss, 0

                self._step_fn = _step

        loss, norm = self._step_fn(batch)

        mx.eval(loss, norm, state)

        return float(loss)

    def train(
        self, dataset: CSMDataset, batch_size: int, epochs: int, shuffle: bool = True
    ) -> History:
        """Train the model on the dataset."""
        num_samples = len(dataset)
        _steps_per_epoch = (num_samples + batch_size - 1) // batch_size

        start_epoch = self.state.epoch
        start_step = self.state.step

        resume_batch_idx = 0
        if start_epoch < epochs and start_step > 0:
            completed_steps_before_epoch = start_epoch * _steps_per_epoch
            if start_step > completed_steps_before_epoch:
                resume_batch_idx = start_step % _steps_per_epoch

        if start_epoch > 0 or resume_batch_idx > 0:
            print(f"Resuming from Epoch {start_epoch + 1}, Step {start_step + 1}")
            print(
                f"(Starting processing from batch index {resume_batch_idx} in epoch {start_epoch + 1})"
            )

        for epoch in range(start_epoch, epochs):
            indices = mx.arange(num_samples)
            if shuffle:
                indices = mx.random.permutation(indices)

            batch_indices = [
                indices[i : i + batch_size].tolist()
                for i in range(0, num_samples, batch_size)
            ]

            epoch_starting_batch_idx = 0
            if epoch == start_epoch:
                epoch_starting_batch_idx = resume_batch_idx

            remaining_batch_indices = batch_indices[epoch_starting_batch_idx:]

            if not remaining_batch_indices:
                print(
                    f"Epoch {epoch + 1} already fully completed in previous run. Skipping."
                )
                self.state.epoch = epoch + 1
                continue

            pbar = tqdm(
                enumerate(remaining_batch_indices, start=epoch_starting_batch_idx),
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=len(batch_indices),
                initial=epoch_starting_batch_idx,
            )

            epoch_loss_sum = 0.0
            num_batches_processed_this_epoch = 0

            for batch_num_in_full_epoch, batch_idx_list in pbar:
                batch = dataset.get_batch(
                    batch_idx_list  # type: ignore
                )

                loss = self.train_step(batch)

                self.state.step += 1
                self.state.learning_rate = float(self.optimizer.learning_rate)

                epoch_loss_sum += loss
                num_batches_processed_this_epoch += 1

                if self.state.step % self.args.log_freq == 0:
                    self.history.log(
                        self.state.step,
                        epoch,
                        loss,
                        self.state.learning_rate,
                    )
                    pbar.set_postfix(
                        {
                            "step": self.state.step,
                            "loss": f"{loss:.4f}",
                            "lr": f"{self.state.learning_rate:.1E}",
                        }
                    )

                if (
                    self.args.ckpt_freq > 0
                    and self.state.step % self.args.ckpt_freq == 0
                ):
                    self.checkpointer.save()

            if num_batches_processed_this_epoch > 0:
                avg_epoch_loss = epoch_loss_sum / num_batches_processed_this_epoch
                print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1} had no batches processed in this run.")

            self.state.epoch = epoch + 1

            print(f"Completed Epoch {epoch + 1}. Saving checkpoint.")
            self.checkpointer.save()

        return self.history


class DPOTrainer(CSMTrainer):
    def __init__(self, args: DPOArgs):
        if not isinstance(args, DPOArgs):
            raise TypeError(
                "Please use `DPOArgs` instead of other trainer's arguments."
            )

        super().__init__(args)
        self.beta = args.beta

    @staticmethod
    def compute_loss(model: CSM, batch: Dict[str, mx.array], **kwargs) -> mx.array:
        beta = batch["beta"]
        fcw = batch["first_codebook_weight_multiplier"]

        chosen = {
            "tokens": batch["chosen_tokens"],
            "masks": batch["chosen_masks"],
            "loss_masks": batch["chosen_loss_masks"],
        }
        rejected = {
            "tokens": batch["rejected_tokens"],
            "masks": batch["rejected_masks"],
            "loss_masks": batch["rejected_loss_masks"],
        }

        chosen_loss = CSMTrainer.compute_loss(
            model,
            {
                **chosen,
                "first_codebook_weight_multiplier": fcw,
            },
            per_sample=True,
        )
        rejected_loss = CSMTrainer.compute_loss(
            model,
            {
                **rejected,
                "first_codebook_weight_multiplier": fcw,
            },
            per_sample=True,
        )

        margin = -(chosen_loss - rejected_loss) * beta
        return mx.mean(-nn.log_sigmoid(margin))

    def train_step(self, batch: Dict[str, mx.array]) -> float:
        model = self.model
        optimizer = self.optimizer
        first_codebook_weight_multiplier = mx.array(
            self.args.first_codebook_weight_multiplier
        )
        beta = mx.array(self.beta, dtype=mx.float32)

        state = [
            model.state,
            optimizer.state,
            mx.random.state,
        ]

        if self._step_fn is None:
            loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)  # type: ignore

            if self.args.max_norm > 0:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "first_codebook_weight_multiplier": first_codebook_weight_multiplier,
                            "beta": beta,
                        },
                    )

                    grads, norm = optim.clip_grad_norm(grads, self.args.max_norm)

                    optimizer.update(model, grads)

                    return loss, norm

                self._step_fn = _step
            else:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "first_codebook_weight_multiplier": first_codebook_weight_multiplier,
                            "beta": beta,
                        },
                    )

                    optimizer.update(model, grads)

                    return loss, 0

                self._step_fn = _step

        loss, norm = self._step_fn(batch)

        mx.eval(loss, norm, state)

        return float(loss)

    def train(
        self,
        dataset: CSMDataset,
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
    ):
        if not isinstance(dataset, CSMPairwiseDataset):
            raise TypeError(
                "Please use `CSMPairwiseDataset` instead of other dataset types."
            )
        return super().train(dataset, batch_size, epochs, shuffle)


class KTOTrainer(CSMTrainer):
    def __init__(self, args: KTOArgs):
        if not isinstance(args, KTOArgs):
            raise TypeError(
                "Please use `KTOArgs` instead of other trainer's arguments."
            )

        if args.reference_model is None:
            raise ValueError("Reference model must be provided.")

        super().__init__(args)
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.reference_model = args.reference_model

        self.reference_model.eval()

    @staticmethod
    def compute_loss(model: CSM, batch: Dict[str, mx.array], **kwargs) -> mx.array:
        beta = batch["beta"]
        fcw = batch["first_codebook_weight_multiplier"]
        desirable_weight = batch["desirable_weight"]
        undesirable_weight = batch["undesirable_weight"]

        preference = batch["preference"]

        kl_reference = batch["kl_reference"]
        kl_policy = batch["kl_policy"]

        reference = batch["reference"]
        policy = CSMTrainer.compute_loss(
            model,
            {
                **batch,
                "first_codebook_weight_multiplier": fcw,
            },
            per_sample=True,
        )

        reward = policy - reference
        kl = mx.clip((kl_policy - kl_reference).mean(), 0, None)

        penalized_reward = reward - kl

        desirable_mask = preference > 0
        desirable_losses = (
            desirable_weight
            * desirable_mask
            * (1 - mx.sigmoid(beta * penalized_reward))
        )

        undesirable_mask = preference < 0
        undesirable_losses = (
            undesirable_weight
            * undesirable_mask
            * (1 - mx.sigmoid(-beta * penalized_reward))
        )

        total_losses = desirable_losses + undesirable_losses

        return total_losses.mean()

    def train_step(self, batch: Dict[str, mx.array]) -> float:  # type: ignore[override]
        model = self.model
        optimizer = self.optimizer
        first_codebook_weight_multiplier = mx.array(
            self.args.first_codebook_weight_multiplier
        )
        beta = mx.array(self.beta, dtype=mx.float32)
        desirable_weight = mx.array(self.desirable_weight, dtype=mx.float32)
        undesirable_weight = mx.array(self.undesirable_weight, dtype=mx.float32)

        state = [
            model.state,
            optimizer.state,
            mx.random.state,
        ]

        batch = {
            **batch,
            "first_codebook_weight_multiplier": first_codebook_weight_multiplier,
        }

        if self._step_fn is None:
            loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)  # type: ignore

            if self.args.max_norm > 0:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    kl_reference = CSMTrainer.compute_loss(
                        self.reference_model,
                        batch,
                        per_sample=True,
                        cause_mismatch=True,
                    )
                    kl_policy = CSMTrainer.compute_loss(
                        model, batch, per_sample=True, cause_mismatch=True
                    )

                    reference = CSMTrainer.compute_loss(
                        self.reference_model,
                        batch,
                        per_sample=True,
                    )
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "kl_reference": kl_reference,
                            "kl_policy": kl_policy,
                            "reference": reference,
                        },
                    )

                    grads, norm = optim.clip_grad_norm(grads, self.args.max_norm)

                    optimizer.update(model, grads)

                    return loss, norm

                self._step_fn = _step
            else:

                @partial(mx.compile, inputs=state, outputs=state)
                def _step(batch: Dict[str, mx.array]):  # type: ignore
                    kl_reference = CSMTrainer.compute_loss(
                        self.reference_model,
                        batch,
                        per_sample=True,
                        cause_mismatch=True,
                    )
                    kl_policy = CSMTrainer.compute_loss(
                        model, batch, per_sample=True, cause_mismatch=True
                    )

                    reference = CSMTrainer.compute_loss(
                        self.reference_model,
                        batch,
                        per_sample=True,
                    )
                    loss, grads = loss_and_grad_fn(
                        self.model,
                        {
                            **batch,
                            "kl_reference": kl_reference,
                            "kl_policy": kl_policy,
                            "reference": reference,
                        },
                    )

                    optimizer.update(model, grads)

                    return loss, 0

                self._step_fn = _step

        loss, norm = self._step_fn(
            {
                **batch,
                "beta": beta,
                "desirable_weight": desirable_weight,
                "undesirable_weight": undesirable_weight,
            }
        )

        mx.eval(loss, norm, state)

        return float(loss)

    def train(
        self,
        dataset: CSMDataset,
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
    ):
        if not isinstance(dataset, CSMPairwiseDataset):
            raise TypeError(
                "Please use `CSMPairwiseDataset` instead of other dataset types."
            )
        return super().train(dataset, batch_size, epochs, shuffle)
