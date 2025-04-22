from functools import cache
from typing import List

import mlx.core as mx
from huggingface_hub import hf_hub_download
from moshi_mlx.models.mimi import Mimi, mimi_202407
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, LlamaTokenizer

from csm_mlx.config import TOKENIZERS
from csm_mlx.segment import Segment


@cache
def get_audio_tokenizer(n_audio_codebooks: int) -> Mimi:
    mimi = Mimi(mimi_202407(n_audio_codebooks))
    weight = hf_hub_download(**TOKENIZERS["audio"])  # type: ignore

    mimi.load_pytorch_weights(weight)

    return mimi


@cache
def get_text_tokenizer() -> LlamaTokenizer:
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS["text"]["repo_id"])
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )
    return tokenizer


def tokenize_text_segment(text: str, speaker: int) -> tuple[mx.array, mx.array]:
    frame_tokens = []
    frame_masks = []

    text_tokenizer = get_text_tokenizer()

    text_tokens = text_tokenizer.encode(f"[{speaker}]{text}")
    text_frame = mx.zeros((len(text_tokens), 33), dtype=mx.int32)
    text_frame_mask = mx.zeros((len(text_tokens), 33), dtype=mx.int32)
    text_frame[:, -1] = mx.array(text_tokens, dtype=mx.int32)
    text_frame_mask[:, -1] = True

    frame_tokens.append(text_frame)
    frame_masks.append(text_frame_mask)

    return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)


def tokenize_audio(
    audio: mx.array, *, n_audio_codebooks: int = 32
) -> tuple[mx.array, mx.array]:
    frame_tokens = []
    frame_masks = []

    audio_tokenizer = get_audio_tokenizer(n_audio_codebooks)

    # (K, T)
    audio_tokens = audio_tokenizer.encode(mx.expand_dims(mx.expand_dims(audio, 0), 0))[
        0
    ]
    # add EOS frame
    eos_frame = mx.zeros((audio_tokens.shape[0], 1))
    audio_tokens = mx.concat([audio_tokens, eos_frame], axis=1)

    audio_frame = mx.zeros((audio_tokens.shape[1], 33), dtype=mx.int32)
    audio_frame_mask = mx.zeros((audio_tokens.shape[1], 33), dtype=mx.int32)
    audio_frame[:, :-1] = audio_tokens.swapaxes(0, 1)
    audio_frame_mask[:, :-1] = True

    frame_tokens.append(audio_frame)
    frame_masks.append(audio_frame_mask)

    return mx.concat(frame_tokens, axis=0), mx.concat(frame_masks, axis=0)


def tokenize_segment(
    segment: Segment, *, n_audio_codebooks: int = 32
) -> tuple[mx.array, mx.array]:
    """
    Returns:
        (seq_len, 33), (seq_len, 33)
    """
    text_tokens, text_masks = tokenize_text_segment(segment.text, segment.speaker)
    audio_tokens, audio_masks = tokenize_audio(
        segment.audio, n_audio_codebooks=n_audio_codebooks
    )

    return mx.concat([text_tokens, audio_tokens], axis=0).astype(mx.int32), mx.concat(
        [text_masks, audio_masks], axis=0
    ).astype(mx.bool_)  # type: ignore


def tokenize_segments_with_loss_mask(
    segments: List[Segment],
    *,
    n_audio_codebooks: int = 32,
    mask_speaker_ids: List[int],
    max_audio_length_ms: int | None,
) -> tuple[mx.array, mx.array, mx.array]:
    tokens_list, masks_list = map(
        list,
        zip(
            *[
                tokenize_segment(
                    segment,
                    n_audio_codebooks=n_audio_codebooks,
                )
                for segment in segments
            ]
        ),
    )

    tokens = mx.concatenate(tokens_list, axis=0)
    masks = mx.concatenate(masks_list, axis=0)
    loss_masks = mx.ones_like(tokens)

    token_position = 0
    for sep_token, segment in zip(tokens_list, segments):
        segment_length = sep_token.shape[0]

        if segment.speaker in mask_speaker_ids:
            loss_masks[token_position : token_position + segment_length] = 0

        token_position += segment_length

    # Apply maximum length constraint if specified
    if max_audio_length_ms is not None:
        max_tokens = int(max_audio_length_ms / 80)
        tokens = tokens[:max_tokens]
        masks = masks[:max_tokens]
        loss_masks = loss_masks[:max_tokens]

    return tokens, masks, loss_masks


def decode_audio(audio_tokens: mx.array, *, n_audio_codebooks=32) -> mx.array:
    audio_tokenizer = get_audio_tokenizer(n_audio_codebooks)
    return audio_tokenizer.decode(audio_tokens)
