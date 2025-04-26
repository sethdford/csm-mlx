import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from csm_mlx.segment import Segment
from csm_mlx.tokenizers import tokenize_segments_with_loss_mask


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
        return tokenize_segments_with_loss_mask(
            self.samples[idx],
            n_audio_codebooks=self.n_audio_codebooks,
            mask_speaker_ids=self.mask_speaker_ids,
            max_audio_length_ms=self.max_audio_length_ms,
        )

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


class CSMPairwiseDataset(CSMDataset):
    """Dataset of paired conversations: each example has a ‘chosen’ and a ‘rejected’ conversations."""

    def __init__(
        self,
        pairs: List[Tuple[List[Segment], List[Segment]]],
        n_audio_codebooks: int = 32,
        max_audio_length_ms: Optional[int] = None,
        mask_speaker_ids: Optional[int | List[int]] = None,
    ):
        self.pairs = pairs
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
    ) -> "CSMPairwiseDataset":
        """
        Load dataset from a JSON file with format:
        [
            {
                "chosen": [
                    {"text": "...", "audio_path": "...", "speaker": 0},
                    ...
                ],
                "rejected": [
                    {"text": "...", "audio_path": "...", "speaker": 1},
                    ...
                ]
            },
            ...
        ]
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        pairs: List[Tuple[List[Segment], List[Segment]]] = []
        for item in data:
            chosen_segs = [
                Segment(
                    text=seg["text"],
                    audio_path=Path(seg["audio_path"]),
                    speaker=seg.get("speaker", 0),
                )
                for seg in item["chosen"]
            ]
            rejected_segs = [
                Segment(
                    text=seg["text"],
                    audio_path=Path(seg["audio_path"]),
                    speaker=seg.get("speaker", 0),
                )
                for seg in item["rejected"]
            ]
            pairs.append((chosen_segs, rejected_segs))

        return cls(pairs, n_audio_codebooks, max_audio_length_ms, mask_speaker_ids)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Tuple[mx.array, mx.array, mx.array]]:  # type: ignore
        chosen_segs, rejected_segs = self.pairs[idx]

        return {
            "chosen": tokenize_segments_with_loss_mask(
                chosen_segs,
                n_audio_codebooks=self.n_audio_codebooks,
                mask_speaker_ids=self.mask_speaker_ids,
                max_audio_length_ms=self.max_audio_length_ms,
            ),
            "rejected": tokenize_segments_with_loss_mask(
                rejected_segs,
                n_audio_codebooks=self.n_audio_codebooks,
                mask_speaker_ids=self.mask_speaker_ids,
                max_audio_length_ms=self.max_audio_length_ms,
            ),
        }

    def get_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        batch = {
            "chosen_tokens": [],
            "rejected_tokens": [],
            "chosen_masks": [],
            "rejected_masks": [],
            "chosen_loss_masks": [],
            "rejected_loss_masks": [],
        }

        for i in indices:
            ex = self[i]
            for key in ("chosen", "rejected"):
                token, mask, loss_mask = ex[key]
                batch[f"{key}_tokens"].append(token)
                batch[f"{key}_masks"].append(mask)
                batch[f"{key}_loss_masks"].append(loss_mask)

        all_lengths = []
        for key in ("chosen", "rejected"):
            all_lengths += [token.shape[0] for token in batch[f"{key}_tokens"]]
        max_len = max(all_lengths)

        out: Dict[str, mx.array] = {}
        for key in ("chosen", "rejected"):
            tokens = batch[f"{key}_tokens"]
            masks = batch[f"{key}_masks"]
            loss_masks = batch[f"{key}_loss_masks"]

            padded_tokens, padded_masks, padded_loss_masks = [], [], []
            for token, mask, loss_mask in zip(tokens, masks, loss_masks):
                pad = max_len - token.shape[0]
                if pad > 0:
                    padded_tokens.append(
                        mx.pad(token, [(0, pad), (0, 0)], constant_values=0)
                    )
                    padded_masks.append(
                        mx.pad(mask, [(0, pad), (0, 0)], constant_values=0)
                    )
                    padded_loss_masks.append(
                        mx.pad(loss_mask, [(0, pad), (0, 0)], constant_values=0)
                    )
                else:
                    padded_tokens.append(token)
                    padded_masks.append(mask)
                    padded_loss_masks.append(loss_mask)

            out[f"{key}_tokens"] = mx.stack(padded_tokens)
            out[f"{key}_masks"] = mx.stack(padded_masks)
            out[f"{key}_loss_masks"] = mx.stack(padded_loss_masks)

        return out


class CSMPointwiseDataset(CSMDataset):
    """Dataset of single conversations + a preference (+1 for chosen, -1 for rejected)."""

    def __init__(
        self,
        entries: List[Tuple[List[Segment], int]],
        n_audio_codebooks: int = 32,
        max_audio_length_ms: Optional[int] = None,
        mask_speaker_ids: Optional[int | List[int]] = None,
    ):
        self.entries = entries
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
    ) -> "CSMPointwiseDataset":
        """
        Load dataset from a JSON file with format:
        [
            {
                "segments": [
                    {"text": "...", "audio_path": "...", "speaker": 0},
                    ...
                ],
                "preference": 1
            },
            ...
        ]
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        entries: List[Tuple[List[Segment], int]] = []
        for item in data:
            segments = [
                Segment(
                    text=seg["text"],
                    audio_path=Path(seg["audio_path"]),
                    speaker=seg.get("speaker", 0),
                )
                for seg in item["segments"]
            ]
            preference = int(item["preference"])
            entries.append((segments, preference))

        return cls(entries, n_audio_codebooks, max_audio_length_ms, mask_speaker_ids)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array, mx.array, int]:  # type: ignore
        segments, label = self.entries[idx]

        return *tokenize_segments_with_loss_mask(
            segments,
            n_audio_codebooks=self.n_audio_codebooks,
            mask_speaker_ids=self.mask_speaker_ids,
            max_audio_length_ms=self.max_audio_length_ms,
        ), label

    def get_batch(self, indices: List[int]) -> Dict[str, mx.array]:
        batch_tokens, batch_masks, batch_loss_masks, batch_preferences = [], [], [], []
        for i in indices:
            token, mask, loss_mask, preference = self[i]
            batch_tokens.append(token)
            batch_masks.append(mask)
            batch_loss_masks.append(loss_mask)
            batch_preferences.append(preference)

        max_len = max(t.shape[0] for t in batch_tokens)
        padded_tokens, padded_masks, padded_loss_masks = [], [], []
        for t, m, lm in zip(batch_tokens, batch_masks, batch_loss_masks):
            pad = max_len - t.shape[0]
            if pad:
                padded_tokens.append(mx.pad(t, [(0, pad), (0, 0)], constant_values=0))
                padded_masks.append(mx.pad(m, [(0, pad), (0, 0)], constant_values=0))
                padded_loss_masks.append(
                    mx.pad(lm, [(0, pad), (0, 0)], constant_values=0)
                )
            else:
                padded_tokens.append(t)
                padded_masks.append(m)
                padded_loss_masks.append(lm)

        return {
            "tokens": mx.stack(padded_tokens),
            "masks": mx.stack(padded_masks),
            "loss_masks": mx.stack(padded_loss_masks),
            "preferences": mx.array(batch_preferences, dtype=mx.int32),
        }
