import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from csm_mlx.segment import Segment
from csm_mlx.tokenizers import tokenize_segment


class CSMDataset:
    """Dataset for finetuning CSM models."""

    def __init__(
        self,
        samples: List[List[Segment]],
        n_audio_codebooks: int = 32,
        max_audio_length_ms: Optional[int] = None,
        mask_speaker_ids: Optional[int | List[int]] = None,
    ):
        self.samples = samples
        self.n_audio_codebooks = n_audio_codebooks
        self.max_audio_length_ms = max_audio_length_ms
        self.mask_speaker_ids = (
            mask_speaker_ids
            if isinstance(mask_speaker_ids, list)
            else [mask_speaker_ids]
            if mask_speaker_ids is not None
            else []
        )

    @classmethod
    def from_json(
        cls,
        json_path: str,
        n_audio_codebooks: int = 32,
        max_audio_length_ms: Optional[int] = None,
        mask_speaker_ids: Optional[int | List[int]] = None,
    ) -> "CSMDataset":
        """Load dataset from a JSON file with format:

        [
            [
                {"text": "Sample text", "audio_path": "/path/to/audio.wav", "speaker": 0},
                ...
            ]
        ]
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Create LazySegment instances for each item
        samples = [
            [
                Segment(
                    text=item["text"],
                    audio_path=Path(item["audio_path"]),
                    speaker=item.get("speaker", 0),
                )
                for item in conversation
            ]
            for conversation in data
        ]

        return cls(
            samples,
            n_audio_codebooks=n_audio_codebooks,
            max_audio_length_ms=max_audio_length_ms,
            mask_speaker_ids=mask_speaker_ids,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array, mx.array]:
        """Get tokenized representation of the sample."""
        sample = self.samples[idx]

        # Tokenize segments (audio will be loaded lazily when needed)
        tokens_list, masks_list = map(
            list,
            zip(
                *[
                    tokenize_segment(
                        segment,
                        n_audio_codebooks=self.n_audio_codebooks,
                    )
                    for segment in sample
                ]
            ),
        )

        tokens = mx.concatenate(tokens_list, axis=0)
        masks = mx.concatenate(masks_list, axis=0)
        loss_masks = mx.ones_like(tokens)

        token_position = 0
        for sep_token, segment in zip(tokens_list, sample):
            segment_length = sep_token.shape[0]

            if segment.speaker in self.mask_speaker_ids:
                loss_masks[token_position : token_position + segment_length] = 0

            token_position += segment_length

        # Apply maximum length constraint if specified
        if self.max_audio_length_ms is not None:
            max_tokens = int(self.max_audio_length_ms / 80)
            tokens = tokens[:max_tokens]
            masks = masks[:max_tokens]
            loss_masks = loss_masks[:max_tokens]

        return tokens, masks, loss_masks

    def get_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        """Get a batch of samples."""
        batch_tokens, batch_masks, batch_loss_masks = [], [], []

        for idx in indices:
            tokens, masks, loss_masks = self[idx]
            batch_tokens.append(tokens)
            batch_masks.append(masks)
            batch_loss_masks.append(loss_masks)

        # Handle variable sequence lengths by padding to the longest in the batch
        max_len = max(tokens.shape[0] for tokens in batch_tokens)

        padded_tokens = []
        padded_masks = []
        padded_loss_masks = []

        for tokens, masks, loss_masks in zip(
            batch_tokens, batch_masks, batch_loss_masks
        ):
            pad_len = max_len - tokens.shape[0]

            if pad_len > 0:
                padded_tokens.append(
                    mx.pad(tokens, [(0, pad_len), (0, 0)], constant_values=0)
                )
                padded_masks.append(
                    mx.pad(masks, [(0, pad_len), (0, 0)], constant_values=0)
                )
                padded_loss_masks.append(
                    mx.pad(loss_masks, [(0, pad_len), (0, 0)], constant_values=0)
                )
            else:
                padded_tokens.append(tokens)
                padded_masks.append(masks)
                padded_loss_masks.append(loss_masks)

        return {
            "tokens": mx.stack(padded_tokens),
            "masks": mx.stack(padded_masks),
            "loss_masks": mx.stack(padded_loss_masks),
        }
