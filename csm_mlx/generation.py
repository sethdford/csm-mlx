from collections.abc import Callable
from typing import Any, Generator, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from csm_mlx.models import CSM
from csm_mlx.segment import Segment
from csm_mlx.tokenizers import (
    decode_audio,
    get_audio_tokenizer,
    tokenize_segment,
    tokenize_text_segment,
)

default_stream = mx.new_stream(mx.default_device())


def generate_frame(
    model: CSM,
    tokens: mx.array,
    *,
    token_mask: Optional[mx.array] = None,
    sampler: Optional[Callable[..., mx.array]] = None,
    cache: Optional[Any] = None,
    stream: mx.Stream = default_stream,
) -> mx.array:
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    token_mask = token_mask if token_mask is not None else mx.ones_like(tokens)

    backbone_embeds = model.embed_tokens(tokens)
    backbone_embeds = backbone_embeds * mx.expand_dims(token_mask, axis=-1)
    backbone_input = backbone_embeds.sum(-2)

    with mx.stream(stream):
        backbone_hidden = model.backbone(backbone_input, cache=cache)
        backbone_last_hidden = backbone_hidden[:, -1, :]

        c0_logits = model.codebook0_head(backbone_last_hidden)
        c0_sample = mx.expand_dims(sampler(c0_logits), axis=-1)
        c0_embeds = model.embed_audio(0, c0_sample)

        decoder_inputs = mx.concat(
            [mx.expand_dims(backbone_last_hidden, axis=1), c0_embeds], axis=1
        )
        decoder_sample = c0_sample

        decoder_cache = make_prompt_cache(model.decoder)
        for index in range(1, model.n_audio_codebooks):
            decoder_hidden = model.decoder(
                model.projection(decoder_inputs),
                cache=decoder_cache,
            )

            ci_logits = mx.matmul(decoder_hidden[:, -1, :], model.audio_head[index - 1])
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
    stream: mx.Stream = default_stream,
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
    input = mx.expand_dims(prompt_tokens, 0)
    mask = mx.expand_dims(prompt_tokens_mask, 0)
    backbone_cache = make_prompt_cache(model.backbone)

    max_seq_len = 2048 - max_audio_frames
    if input.shape[1] >= max_seq_len:
        raise ValueError(
            f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
        )

    for _ in range(max_audio_frames):
        sample = generate_frame(
            model,
            input,
            sampler=sampler,
            token_mask=mask,
            cache=backbone_cache,
            stream=stream,
        )

        if not sample.any():
            break  # eos

        samples.append(sample)

        input = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1))], axis=1), 1).astype(
            mx.int64
        )
        mask = mx.expand_dims(
            mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
        ).astype(mx.bool_)  # type: ignore

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


def stream_generate(
    model: CSM,
    text: str,
    speaker: int,
    context: list[Segment],
    max_audio_length_ms: float = 90_000,
    *,
    sampler: Optional[Callable[..., mx.array]] = None,
    stream: mx.Stream = default_stream,
) -> Generator[mx.array, None, None]:
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

    input = mx.expand_dims(prompt_tokens, 0)
    mask = mx.expand_dims(prompt_tokens_mask, 0)
    backbone_cache = make_prompt_cache(model.backbone)

    max_seq_len = 2048 - max_audio_frames
    if input.shape[1] >= max_seq_len:
        raise ValueError(
            f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
        )

    audio_tokenizer = get_audio_tokenizer()
    audio_tokenizer.reset_state()

    for _ in range(max_audio_frames):
        sample = generate_frame(
            model,
            input,
            sampler=sampler,
            token_mask=mask,
            cache=backbone_cache,
            stream=stream,
        )

        if not sample.any():
            break  # eos

        input = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1))], axis=1), 1).astype(
            mx.int64
        )
        mask = mx.expand_dims(
            mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
        )

        yield audio_tokenizer.decode_step(mx.expand_dims(sample.transpose(1, 0), 2))
