"""fp16-accumulate flash attention for vit.sdpa: the vendored fastbb kernel (head_dim 64, any N); falls back to torch SDPA
inside the kernel wrapper for anything it does not cover."""
from ..fastbb_vendored.attention import attention_bhnd


def attention(q, k, v, variant="hh"):
    """q, k, v [B, H, N, 64] (SDPA layout, strided views ok) -> [B, H, N, 64]."""
    return attention_bhnd(q, k, v, variant=variant)
