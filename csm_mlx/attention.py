import math
from typing import Any, Optional

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.llama import ModelArgs


class Llama3ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Default scaling factors are from the following Meta-Llama code:
    https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1/api/model.py#L41

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (int): scaling factor for theta. Default: 8
        low_freq_factor (int): low frequency factor for scaling theta. Default: 1
        high_freq_factor (int): high frequency factor for scaling theta. Default: 4
        old_context_len (int): old context length for scaling theta. Default: 8192
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10_000.0,
        scale_factor: float = 8.0,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.dim, 2)[: (self.dim // 2)].astype(mx.float32)
                / self.dim
            )
        )

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self._theta = theta
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = mx.arange(max_seq_len, dtype=self._theta.dtype)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = mx.einsum("i, j -> ij", seq_idx, self._theta).astype(mx.float32)

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = mx.stack([mx.cos(idx_theta), mx.sin(idx_theta)], axis=-1)
        self._cache = cache

    def apply_scaling(
        self,
        freqs: mx.array,
        scale_factor: float,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return mx.array(new_freqs, dtype=freqs.dtype)

    def __call__(self, x: mx.array, *, offset: int) -> mx.array:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises:
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """

        if not self.is_cache_built:
            raise RuntimeError(
                "RoPE cache is not built. Please call rope_init() first."
            )

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.shape[1]

        # extract the values based on whether input_pos is set or not
        rope_cache = self._cache[None, offset : offset + seq_len]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.reshape(-1, xshaped.shape[1], 1, xshaped.shape[3], 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = mx.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = Llama3ScaledRoPE(
            self.head_dim,
            base=args.rope_theta,
            scale_factor=args.rope_scaling.get("factor", 1.0),  # type: ignore
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        b, s_x, _ = x.shape

        # q, k, v has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # number of queries per key/value
        q = q.reshape(b, s_x, -1, self.head_dim)
        k = k.reshape(b, s_x, -1, self.head_dim)
        v = v.reshape(b, s_x, -1, self.head_dim)

        # Apply positional embeddings
        if self.rope is not None:
            q = self.rope(q, offset=cache.offset if cache else 0)
            k = self.rope(k, offset=cache.offset if cache else 0)

        # [b, n_h, s_x, h_d]
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        # Update key-value cache
        if cache:
            k, v = cache.update_and_fetch(k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.n_heads != self.n_kv_heads:
            q_per_kv = self.n_heads // self.n_kv_heads

            k = mx.expand_dims(k, axis=2)
            v = mx.expand_dims(v, axis=2)

            k_expand_shape = (b, self.n_kv_heads, q_per_kv) + k.shape[3:]
            v_expand_shape = (b, self.n_kv_heads, q_per_kv) + v.shape[3:]

            k = mx.broadcast_to(k, k_expand_shape)
            v = mx.broadcast_to(v, v_expand_shape)

            k = k.reshape(b, self.n_kv_heads * q_per_kv, *k.shape[3:])
            v = v.reshape(b, self.n_kv_heads * q_per_kv, *v.shape[3:])

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )

        # reshape the output to be the same shape as the input
        output = output.swapaxes(1, 2).reshape(b, s_x, -1)
        return self.o_proj(output)
