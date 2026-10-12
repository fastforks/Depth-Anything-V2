"""Tests for the portable fast path (SDPA attention, dynamic positional embedding, out_conv before upsample, infer_tensor,
prepare_half). Parity is checked against the unmodified math of upstream `infer_image` (image -> cv2 cubic resize -> fp32
network -> bilinear resize back).

    DAV2_CHECKPOINT=/path/depth_anything_v2_vits.pth DAV2_ENCODER=vits python tests/test_fast.py     (or pytest tests/)

Tests marked "needs weights" are skipped without a CUDA GPU or the checkpoint; the others run on the CPU.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from depth_anything_v2.dinov2 import DinoVisionTransformer, vit_small  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

try:
    import pytest
    skip = pytest.skip
except ImportError:  # plain `python tests/test_fast.py`
    class _Skip(Exception):
        pass

    def skip(msg):
        raise _Skip(msg)

CFG = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
       "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
       "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])}
ENC = os.environ.get("DAV2_ENCODER", "vits")
CKPT = os.environ.get("DAV2_CHECKPOINT", "")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _model():
    if not torch.cuda.is_available() or not os.path.exists(CKPT):
        skip("needs a CUDA GPU and DAV2_CHECKPOINT")
    m = DepthAnythingV2(**CFG[ENC])
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    return m.cuda().eval()


def _images():
    """BGR uint8 images (as cv2.imread): a real photo at its size, odd crops / sizes, a constant image."""
    base = cv2.imread(os.path.join(ROOT, "assets", "examples", "demo01.jpg"))
    rng = np.random.RandomState(0)
    return {"photo": base,
            "odd_333x517": np.ascontiguousarray(base[:333, :517]),
            "tall_701x299": np.ascontiguousarray(cv2.resize(base, (299, 701))),
            "small_97x131": np.ascontiguousarray(cv2.resize(base, (131, 97))),
            "constant_128": np.full((300, 400, 3), 128, np.uint8),
            "noise": rng.randint(0, 256, (260, 350, 3)).astype(np.uint8)}


def _rgb_chw(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1).cuda()


def _rel(a, b):
    return ((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)).item()


# --------------------------------------------------------------------------------------- weight-free (CPU) tests
def test_sdpa_attention_matches_explicit():
    """Attention.forward with F.scaled_dot_product_attention = the original explicit softmax(q k^T * scale) v."""
    from depth_anything_v2.dinov2_layers.attention import Attention
    torch.manual_seed(0)
    a = Attention(64, num_heads=4).eval()
    x = torch.randn(2, 37, 64)
    B, N, C = x.shape
    qkv = a.qkv(x).reshape(B, N, 3, 4, C // 4).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * a.scale, qkv[1], qkv[2]
    ref = a.proj(((q @ k.transpose(-2, -1)).softmax(-1) @ v).transpose(1, 2).reshape(B, N, C))
    assert torch.allclose(a(x), ref, atol=1e-6, rtol=1e-5)


def _ref_interpolate(vit, x, w, h):
    """upstream DinoVisionTransformer.interpolate_pos_encoding (F.interpolate bicubic with scale_factor)."""
    npatch, N = x.shape[1] - 1, vit.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return vit.pos_embed
    pos = vit.pos_embed.float()
    cls, patch = pos[:, 0], pos[:, 1:]
    dim = x.shape[-1]
    w0, h0 = w // vit.patch_size, h // vit.patch_size
    w0, h0 = w0 + vit.interpolate_offset, h0 + vit.interpolate_offset
    sq = N ** 0.5
    p = F.interpolate(patch.reshape(1, int(sq), int(sq), dim).permute(0, 3, 1, 2), scale_factor=(float(w0 / sq), float(h0 / sq)),
                      mode="bicubic", antialias=vit.interpolate_antialias)
    p = p.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((cls.unsqueeze(0), p), dim=1).to(x.dtype)


def test_pos_embed_matrices_match_interpolate():
    """Position embedding as two interpolation matrices = F.interpolate(scale_factor, bicubic), for many token grids."""
    torch.manual_seed(0)
    vit = vit_small(img_size=518, patch_size=14, init_values=1.0, block_chunks=0).eval()
    vit.pos_embed.data = torch.randn_like(vit.pos_embed)
    for (h, w) in [(518, 518), (224, 224), (518, 700), (700, 518), (336, 462), (98, 126), (1008, 756), (14, 28), (1022, 686)]:
        x = torch.zeros(1, (h // 14) * (w // 14) + 1, vit.embed_dim)
        got, ref = vit.interpolate_pos_encoding(x, w, h), _ref_interpolate(vit, x, w, h)
        assert got.shape == ref.shape, (h, w, got.shape, ref.shape)
        assert (got - ref).abs().max().item() < 2e-5, (h, w, (got - ref).abs().max().item())


def test_out_conv_commutes_with_upsample():
    """FeatureFusionBlock: out_conv (1x1) then bilinear upsample = bilinear upsample then out_conv (original order)."""
    from depth_anything_v2.util.blocks import FeatureFusionBlock
    torch.manual_seed(0)
    blk = FeatureFusionBlock(32, torch.nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True).eval()
    x0, x1 = torch.randn(1, 32, 12, 17), torch.randn(1, 32, 12, 17)
    got = blk(x0, x1)
    out = blk.resConfUnit1(x1) + x0
    out = blk.resConfUnit2(out)
    ref = blk.out_conv(F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True))
    assert got.shape == ref.shape and (got - ref).abs().max().item() < 1e-5


def test_pos_embed_torch_export_dynamic():
    """One torch.export trace with a dynamic token grid is exact for other grids (no Python float of the grid size, no einsum)."""
    vit = vit_small(img_size=518, patch_size=14, init_values=1.0, block_chunks=0).eval()
    vit.pos_embed.data = torch.randn_like(vit.pos_embed)

    class PosEmbed(torch.nn.Module):
        def forward(self, grid):  # grid: [1, 3, gh, gw] token-grid carrier
            gh, gw = grid.shape[-2:]
            return vit.interpolate_pos_encoding(torch.zeros(1, gh * gw + 1, vit.embed_dim), gw * 14, gh * 14)

    dh, dw = torch.export.Dim("gh", min=2, max=100), torch.export.Dim("gw", min=2, max=100)
    ep = torch.export.export(PosEmbed(), (torch.zeros(1, 3, 19, 28),), dynamic_shapes={"grid": {2: dh, 3: dw}})
    for gh, gw in [(19, 28), (37, 20), (50, 50), (37, 37)]:
        g = torch.zeros(1, 3, gh, gw)
        assert (ep.module()(g) - PosEmbed()(g)).abs().max().item() < 1e-6
    assert torch.equal(ep.module()(torch.zeros(1, 3, 37, 37)), vit.pos_embed)   # native grid: identity, as eager


def test_fp32_head_conv_overrides_autocast():
    from depth_anything_v2.dpt import _Float32
    layer = _Float32(torch.nn.Conv2d(3, 1, 1))
    x = torch.randn(1, 3, 5, 7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)
    assert out.dtype == torch.float32 and torch.equal(out, layer(x))


# --------------------------------------------------------------------------------------- needs weights
def test_infer_tensor_fp32_parity():
    m = _model()
    for name, bgr in _images().items():
        ref = m.infer_image(bgr, 518)
        got = m.infer_tensor(_rgb_chw(bgr), 518).cpu().numpy()
        assert got.shape == ref.shape and np.isfinite(got).all(), name
        rel = float(np.abs(got - ref).mean() / max(np.abs(ref).mean(), 1e-6))
        assert rel < 2e-4, (name, rel)   # measured 3e-5 .. 5e-5 (cv2 vs torch bicubic rounding)


def test_infer_tensor_other_input_sizes():
    m = _model()
    bgr = _images()["photo"]
    for isz in (266, 518, 700):
        ref = m.infer_image(bgr, isz)
        got = m.infer_tensor(_rgb_chw(bgr), isz).cpu().numpy()
        assert got.shape == ref.shape
        assert float(np.abs(got - ref).mean() / np.abs(ref).mean()) < 2e-4, isz


def test_prepare_half_parity_and_finite():
    m = _model()
    imgs = _images()
    refs = {k: m.infer_image(v, 518) for k, v in imgs.items()}
    m.prepare_half()
    assert next(m.parameters()).dtype == torch.float16
    assert m.depth_head.scratch.output_conv2[2].m.weight.dtype == torch.float32   # last 1x1 conv stays fp32
    for name, bgr in imgs.items():
        got = m.infer_tensor(_rgb_chw(bgr), 518).cpu().numpy()
        assert np.isfinite(got).all(), name
        rel = float(np.abs(got - refs[name]).mean() / max(np.abs(refs[name]).mean(), 1e-6))
        assert rel < 4e-3, (name, rel)   # measured 4e-4 .. 8e-4 (fp16)
    got = m.infer_image(imgs["photo"], 518)   # numpy path after prepare_half
    assert float(np.abs(got - refs["photo"]).mean() / np.abs(refs["photo"]).mean()) < 4e-3


def test_constant_image_is_finite_and_close():
    m = _model()
    bgr = _images()["constant_128"]
    ref = m.infer_image(bgr, 518)
    out = m.infer_tensor(_rgb_chw(bgr), 518).cpu().numpy()
    assert np.isfinite(out).all() and out.shape == ref.shape
    assert float(np.abs(out - ref).max()) < 2e-3 * max(float(np.abs(ref).max()), 1.0)   # measured 2e-4


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("_Skip", "Skipped"):
                print("SKIP", name, e)
            else:
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
