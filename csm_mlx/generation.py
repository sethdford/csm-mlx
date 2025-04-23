from collections.abc import Callable
from typing import Any, Generator, List, Optional
import time

import mlx.core as mx
# import mlx.nn as nn # Removed unused import
from mlx_lm.models.cache import KVCache # Assuming KVCache is the right type
# from mlx_lm.sample import sample as sample_fn # Removed incorrect import

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
    temperature: float = 0.8,
    token_mask: Optional[mx.array] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    cache: Optional[Any] = None,
    stream: mx.Stream = default_stream,
    c0_history: Optional[list[mx.array]] = None,  # Required if doing logit processing
) -> mx.array:
    token_mask = token_mask if token_mask is not None else mx.ones_like(tokens)

    backbone_embeds = model.embed_tokens(tokens)
    backbone_embeds = backbone_embeds * mx.expand_dims(token_mask, axis=-1)
    backbone_input = backbone_embeds.sum(-2)

    with mx.stream(stream):
        backbone_hidden = model.backbone(backbone_input, cache=cache)
        backbone_last_hidden = backbone_hidden[:, -1, :]

        c0_logits = model.codebook0_head(backbone_last_hidden)

        if logits_processors:
            for processor in logits_processors:
                c0_logits = processor(
                    mx.stack(c0_history, 0) if c0_history else mx.zeros((0)),
                    c0_logits,
                )

        if temperature == 0:
            c0_sample = mx.argmax(c0_logits, axis=-1)
        else:
            c0_sample = mx.random.categorical(c0_logits * (1 / temperature))

        c0_sample = mx.expand_dims(c0_sample, axis=-1)
        c0_embeds = model.embed_audio(0, c0_sample)

        if c0_history is not None:
            c0_history.append(c0_sample)

        decoder_inputs = mx.concat(
            [mx.expand_dims(backbone_last_hidden, axis=1), c0_embeds], axis=1
        )
        decoder_sample = mx.zeros(
            (tokens.shape[0], model.n_audio_codebooks), dtype=mx.int32
        )
        decoder_sample[:, :1] = c0_sample.astype(mx.int32)

        decoder_cache = [KVCache() for i in range(len(model.decoder.layers))]

        for index in range(1, model.n_audio_codebooks):
            # Decoder likely only returns hidden state, cache is updated in-place?
            decoder_hidden = model.decoder(
                model.projection(decoder_inputs),
                cache=decoder_cache,
            )

            ci_logits = mx.matmul(decoder_hidden[:, -1, :], model.audio_head[index - 1])

            if temperature == 0:
                ci_sample = mx.argmax(ci_logits, axis=-1)
            else:
                ci_sample = mx.random.categorical(ci_logits * (1 / temperature))

            ci_sample = mx.expand_dims(ci_sample, axis=-1)
            ci_embeds = model.embed_audio(index, ci_sample)

            decoder_inputs = ci_embeds
            decoder_sample[:, index : index + 1] = ci_sample.astype(mx.int32)

    return decoder_sample


def generate(
    model: CSM,
    text: str,
    speaker: int,
    context: list[Segment],
    max_audio_length_ms: float = 90_000,
    *,
    temperature: float = 0.8,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
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

    prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
    prompt_tokens_mask = mx.concat(tokens_mask, axis=0)

    samples = []
    input = mx.expand_dims(prompt_tokens, 0)
    mask = mx.expand_dims(prompt_tokens_mask, 0)

    backbone_cache = [KVCache() for i in range(len(model.backbone.layers))]

    c0_history = []

    # Use model.backbone.args.max_position_embeddings with fallback
    context_window = model.backbone.args.max_position_embeddings or 2048 # Fallback to 2048
    max_seq_len = context_window - max_audio_frames
    if input.shape[1] >= max_seq_len:
        raise ValueError(
            f"Inputs too long ({input.shape[1]}), must be below max_seq_len - max_audio_frames: {max_seq_len}"
        )

    for _ in range(max_audio_frames):
        sample = generate_frame(
            model,
            input,
            temperature=temperature,
            logits_processors=logits_processors,
            token_mask=mask,
            cache=backbone_cache,
            stream=stream,
            c0_history=c0_history,
        )

        if not sample.any():
            break  # eos

        samples.append(sample)

        input = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1))], axis=1), 1).astype(
            mx.int32
        )
        mask = mx.expand_dims(
            mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
        ).astype(mx.bool_)  # type: ignore

    if not samples:
        print("[WARN] No samples generated.")
        return mx.zeros((0,), dtype=mx.float32)

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
    temperature: float = 0.8,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
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

    prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
    prompt_tokens_mask = mx.concat(tokens_mask, axis=0)

    input = mx.expand_dims(prompt_tokens, 0)
    mask = mx.expand_dims(prompt_tokens_mask, 0)

    backbone_cache = [KVCache() for i in range(len(model.backbone.layers))]

    c0_history = []

    # Use model.backbone.args.max_position_embeddings with fallback
    context_window = model.backbone.args.max_position_embeddings or 2048 # Fallback to 2048
    max_seq_len = context_window - max_audio_frames
    if input.shape[1] >= max_seq_len:
        raise ValueError(
            f"Inputs too long ({input.shape[1]}), must be below max_seq_len - max_audio_frames: {max_seq_len}"
        )

    audio_tokenizer = get_audio_tokenizer(n_audio_codebooks=model.n_audio_codebooks)
    audio_tokenizer.reset_state()

    for _ in range(max_audio_frames):
        sample = generate_frame(
            model,
            input,
            temperature=temperature,
            logits_processors=logits_processors,
            token_mask=mask,
            cache=backbone_cache,
            stream=stream,
            c0_history=c0_history,
        )

        if not sample.any():
            break  # eos

        input = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1))], axis=1), 1).astype(
            mx.int32
        )
        mask = mx.expand_dims(
            mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
        ).astype(mx.bool_)  # type: ignore

        with mx.stream(stream):
            yield (
                audio_tokenizer.decode_step(
                    mx.expand_dims(sample, 0).transpose(1, 2, 0)
                )
                .squeeze(0)
                .squeeze(0)
            )

    audio_tokenizer.reset_state()
