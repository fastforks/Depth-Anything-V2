"""Tests for the custom-kernel path (FastDAv2 and the vendored kernels).

    DAV2_CHECKPOINT=/path/depth_anything_v2_vits.pth DAV2_ENCODER=vits python tests/test_kernels.py     (or pytest tests/)

All tests need a CUDA GPU; the model tests also need the checkpoint (skipped otherwise).
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fast import CFG, ENC, CKPT, _images, _rgb_chw, skip  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def _need_gpu():
    if not torch.cuda.is_available():
        skip("needs a CUDA GPU")


def _load():
    _need_gpu()
    if not os.path.exists(CKPT):
        skip("needs DAV2_CHECKPOINT")
    m = DepthAnythingV2(**CFG[ENC])
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    return m.cuda().eval()


def test_flash_attention_matches_sdpa():
    _need_gpu()
    from depth_anything_v2.fastbb_vendored.attention import attention_bhnd
    torch.manual_seed(0)
    for N in (37, 1369, 2701):
        q, k, v = (torch.randn(1, 6, N, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v).float()
        for variant, tol in (("ff", 2e-3), ("fh", 3e-3), ("hh", 1e-2)):
            got = attention_bhnd(q, k, v, variant=variant).float()
            assert (got - ref).abs().max().item() < tol, (N, variant, (got - ref).abs().max().item())


def test_resize_normalize_matches_cv2():
    _need_gpu()
    from depth_anything_v2.mono_vendored.resize import resize_normalize_u8
    from depth_anything_v2.fast_kernels import MEAN, STD
    img = _images()["photo"]
    rgb = np.ascontiguousarray(img[..., ::-1])
    for (h, w) in [(518, 812), (294, 336), (1036, 1554)]:
        ref = cv2.resize(rgb.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC)
        ref = (ref - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        got = resize_normalize_u8(torch.from_numpy(rgb).permute(2, 0, 1).cuda(), h, w, "cubic", MEAN.cuda(), STD.cuda())
        got = got[0].permute(1, 2, 0).float().cpu().numpy()
        assert got.shape == ref.shape and np.abs(got - ref).mean() < 2e-3, (h, w, np.abs(got - ref).mean())


def test_bilinear_kernel_matches_interpolate():
    _need_gpu()
    from depth_anything_v2.mono_vendored import ops
    torch.manual_seed(0)
    x = torch.randn(1, 64, 37, 53, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    for size, ac in (((74, 106), True), ((74, 106), False), ((111, 150), True)):
        ref = F.interpolate(x, size=size, mode="bilinear", align_corners=ac)
        got = ops.interp_bilinear(x, size, ac)
        assert (got.float() - ref.float()).abs().max().item() < 2e-3, (size, ac)


def _rel(a, b):
    return float(np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-6))


def test_fastdav2_matches_infer_image():
    from depth_anything_v2.fast_kernels import FastDAv2
    m = _load()
    imgs = _images()
    refs = {k: m.infer_image(v, 518) for k, v in imgs.items()}
    fast = FastDAv2(m)
    for name, bgr in imgs.items():
        got = fast.infer(_rgb_chw(bgr), 518).cpu().numpy()
        assert got.shape == refs[name].shape and np.isfinite(got).all(), name
        assert _rel(got, refs[name]) < 1.5e-2, (name, _rel(got, refs[name]))   # measured 6e-4 .. 1.8e-3 (ViT-S), up to 5.5e-3 (ViT-L)


def test_fastdav2_kernels_off_and_attention_variants():
    from depth_anything_v2.fast_kernels import FastDAv2
    bgr = _images()["odd_333x517"]
    outs = {}
    for tag, kw in (("hh", {}), ("sdpa", dict(attn=None)), ("nokern", dict(kernels=False)), ("gacc_off", dict(gacc=False))):
        m = _load()
        outs[tag] = FastDAv2(m, **kw).infer(_rgb_chw(bgr), 518).cpu().numpy()
        del m
    for tag, o in outs.items():
        assert np.isfinite(o).all() and _rel(o, outs["nokern"]) < 1.5e-2, (tag, _rel(o, outs["nokern"]))


def test_fastdav2_new_sizes_and_repeat():
    from depth_anything_v2.fast_kernels import FastDAv2
    m = _load()
    bgr = _images()["photo"]
    fast = FastDAv2(m)
    x = _rgb_chw(bgr)
    a = fast.infer(x, 518).cpu().numpy()
    for isz in (266, 700, 1022):
        o = fast.infer(x, isz).cpu().numpy()
        assert o.shape == a.shape and np.isfinite(o).all()
    b = fast.infer(x, 518).cpu().numpy()
    assert np.array_equal(a, b)   # deterministic, cached tables do not change results


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
