# vendored from experiments/004-mono-depth: mono_fast/vit.py @ 2171802 (master c803da8) -- do not edit here
"""General fast inference for DINOv2-style ViTs (the vendored copies in MoGe, DA3, DAv2, UniK3D, UniDepth, Metric3D, ...).

Any input size (multiple of the patch size), nothing compiled or tuned per size:
- pure fp16 weights + activations, attention through torch SDPA (flash kernel; fp32 attention never materialized)
- optional fp16-accumulate cuBLAS GEMMs (`torch.backends.cuda.matmul.allow_fp16_accumulation`, 2x tensor-core rate on GeForce)
- residual + LayerScale as one in-place addcmul per branch
- the positional-embedding interpolation (bicubic, fp32, upstream formula incl. interpolate_offset) cached per token grid
The block math is the upstream DINOv2 Block (eval, no drop path); only the precision / kernel choice changes.
"""
import contextlib
import math

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


ATTN = {"kernel": None}  # None: torch SDPA (flash); "hh" / "fh" / "ff": mono_fast fp16-acc flash kernel (ops.attention)


def sdpa(q, k, v):
    """softmax(q k^T / sqrt(d)) v for fp16 [B, H, N, d]; the custom fp16-accumulate kernel when enabled and d == 64."""
    if ATTN["kernel"] and q.shape[-1] == 64 and q.dtype == torch.float16 and q.is_cuda:
        from .attention import attention  # exp 006 fastbb flash kernel (fp16-accumulate variants)
        return attention(q, k, v, variant=ATTN["kernel"])
    return F.scaled_dot_product_attention(q, k, v)


@contextlib.contextmanager
def attn_kernel(mode):
    old = ATTN["kernel"]
    ATTN["kernel"] = mode
    try:
        yield
    finally:
        ATTN["kernel"] = old


def with_attn(obj, name, kernel):
    """Run obj.<name> with the fastbb flash kernel variant `kernel` ("hh" / "fh" / "ff"; None = torch SDPA), per instance."""
    if not kernel:
        return
    f = getattr(obj, name)

    def wrapped(*a, **k):
        with attn_kernel(kernel):
            return f(*a, **k)
    setattr(obj, name, wrapped)


class FastViT:
    """Wraps a DinoVisionTransformer-like module `vit` (patch_embed, cls_token, pos_embed, register_tokens, blocks, norm)."""

    def __init__(self, vit, dtype=torch.float16, gacc=True):
        self.vit = vit.to(dtype)
        self.vit.eval()  # (UniK3D's DinoVisionTransformer.train() returns None)
        self.dtype = dtype
        self.gacc = gacc
        self.ps = vit.patch_size if isinstance(vit.patch_size, int) else vit.patch_size[0]
        self.nreg = getattr(vit, "num_register_tokens", 0) or 0
        self._pos = {}
        blocks = []
        for b in vit.blocks:
            blocks.extend(list(b) if isinstance(b, torch.nn.ModuleList) or b.__class__.__name__ == "BlockChunk" else [b])
        self.blocks = [b for b in blocks if not isinstance(b, torch.nn.Identity)]
        self.pos_fp32 = vit.pos_embed.detach().float()

    def pos_embed(self, h0, w0):
        key = (h0, w0)
        if key not in self._pos:
            vit = self.vit
            pe = self.pos_fp32
            N = pe.shape[1] - 1
            M = int(math.sqrt(N))
            if N == h0 * w0 and h0 == w0:
                out = pe
            else:
                off = getattr(vit, "interpolate_offset", 0.1)
                kw = dict(scale_factor=((h0 + off) / M, (w0 + off) / M)) if off else dict(size=(h0, w0))
                p = F.interpolate(pe[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2), mode="bicubic",
                                  antialias=getattr(vit, "interpolate_antialias", False), **kw)
                assert p.shape[-2:] == (h0, w0)
                out = torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[key] = out.to(self.dtype)
        return self._pos[key]

    def tokens(self, x):
        vit = self.vit
        B, _, h, w = x.shape
        t = vit.patch_embed(x.to(self.dtype))
        t = torch.cat([vit.cls_token.to(self.dtype).expand(B, -1, -1), t], 1)
        t = t + self.pos_embed(h // self.ps, w // self.ps)
        if self.nreg:
            t = torch.cat([t[:, :1], vit.register_tokens.to(self.dtype).expand(B, -1, -1), t[:, 1:]], 1)
        return t

    @staticmethod
    def block(b, x):
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = F.linear(b.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = sdpa(q, k, v)
        o = F.linear(o.transpose(1, 2).reshape(B, N, C), a.proj.weight, a.proj.bias)
        g1 = b.ls1.gamma if hasattr(b.ls1, "gamma") else None
        x = torch.addcmul(x, o, g1) if g1 is not None else x + o
        m = b.mlp
        hdn = F.linear(b.norm2(x), m.fc1.weight, m.fc1.bias)
        hdn = F.gelu(hdn)
        o = F.linear(hdn, m.fc2.weight, m.fc2.bias)
        g2 = b.ls2.gamma if hasattr(b.ls2, "gamma") else None
        return x.addcmul_(o, g2) if g2 is not None else x.add_(o)

    def intermediate(self, x, layers, norm=True):
        """-> list of (patch tokens [B, N, C] fp16, cls token [B, C]) after the given block indices (normed)."""
        out = []
        with fp16_accumulate(self.gacc):
            t = self.tokens(x)
            for i, b in enumerate(self.blocks):
                t = self.block(b, t)
                if i in layers:
                    o = self.vit.norm(t) if norm else t.clone()
                    out.append((o[:, 1 + self.nreg:], o[:, 0]))
        return out
