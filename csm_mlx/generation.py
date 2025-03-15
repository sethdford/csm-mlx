from collections.abc import Callable
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from csm_mlx.models import CSM
from csm_mlx.segment import Segment
from csm_mlx.tokenizers import (
    decode_audio,
    tokenize_segment,
    tokenize_text_segment,
)

generation_stream = mx.new_stream(mx.default_device())


def generate_frame(
    model: CSM,
    tokens: mx.array,
    *,
    token_mask: Optional[mx.array] = None,
    sampler: Optional[Callable[..., mx.array]] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    token_mask = token_mask if token_mask is not None else mx.ones_like(tokens)

    backbone_embeds = model.embed_tokens(tokens)
    backbone_embeds = backbone_embeds * mx.expand_dims(token_mask, axis=-1)
    backbone_embeds = backbone_embeds.sum(-2)

    with mx.stream(generation_stream):
        hidden = model.backbone(backbone_embeds, cache=cache)
        last_hidden = hidden[:, -1, :]

        c0_logits = model.codebook0_head(last_hidden)
        c0_sample = mx.expand_dims(sampler(c0_logits), axis=-1)
        c0_embeds = model.embed_audio(0, c0_sample)

        decoder_inputs = mx.concat(
            [mx.expand_dims(last_hidden, axis=1), c0_embeds], axis=1
        )
        decoder_sample = c0_sample

        decoder_cache = make_prompt_cache(model.decoder)
        for index in range(1, model.n_audio_codebooks):
            activation = model.decoder(
                model.projection(decoder_inputs),
                cache=decoder_cache,
            )

            ci_logits = mx.matmul(activation[:, -1, :], model.audio_head[index - 1])
            ci_sample = mx.expand_dims(sampler(ci_logits), axis=-1)
            ci_embeds = model.embed_audio(index, ci_sample)

            decoder_inputs = ci_embeds
            decoder_sample = mx.concat([decoder_sample, ci_sample], axis=1)

    return decoder_sample


def generate(
    model: CSM,
    text: str,
    speaker: int,
    context: list[Segment],
    max_audio_length_ms: float = 90_000,
    *,
    sampler: Optional[Callable[..., mx.array]] = None,
) -> mx.array:
    max_audio_frames = int(max_audio_length_ms / 80)

    tokens, tokens_mask = [], []
    for segment in context:
        segment_tokens, segment_tokens_mask = tokenize_segment(
            segment, n_audio_codebooks=model.n_audio_codebooks
        )
        tokens.append(segment_tokens)
        tokens_mask.append(segment_tokens_mask)

    text_segment_tokens, text_segment_tokens_mask = tokenize_text_segment(text, speaker)
    tokens.append(text_segment_tokens)
    tokens_mask.append(text_segment_tokens_mask)

    prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int64)
    prompt_tokens_mask = mx.concat(tokens_mask, axis=0)
    samples = []
    sampled_tokens = mx.expand_dims(prompt_tokens, 0)
    mask = mx.expand_dims(prompt_tokens_mask, 0)
    backbone_cache = make_prompt_cache(model.backbone)

    max_seq_len = 2048 - max_audio_frames
    if sampled_tokens.shape[1] >= max_seq_len:
        raise ValueError(
            f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
        )

    for _ in range(max_audio_frames):
        sample = generate_frame(
            model,
            sampled_tokens,
            sampler=sampler,
            token_mask=mask,
            cache=backbone_cache,
        )

        if sampled_tokens.sum() == 0:
            break  # eos

        samples.append(sample)

        sampled_tokens = mx.expand_dims(
            mx.concat([sample, mx.zeros((1, 1))], axis=1), 1
        ).astype(mx.int64)
        mask = mx.expand_dims(
            mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
        )

    audio = (
        decode_audio(
            mx.stack(samples).transpose(1, 2, 0),
            n_audio_codebooks=model.n_audio_codebooks,
        )
        .squeeze(0)
        .squeeze(0)
    )

    # TODO: Implement watermarking!

    return audio
