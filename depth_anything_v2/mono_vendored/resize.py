# vendored from experiments/004-mono-depth: mono_fast/resize.py @ 2171802 (master c803da8) -- do not edit here
"""GPU image resizes that reproduce OpenCV's cv2.resize (INTER_AREA downscale, INTER_CUBIC upscale, INTER_LINEAR) on uint8
images, as two separable GEMMs with per-size weight matrices (built in float64 on the CPU once per size pair; no tuning).

OpenCV semantics: pixel centres at (i + 0.5); INTER_AREA for downscale = fractional box overlap weights; INTER_CUBIC =
Keys a = -0.75 with replicated borders; INTER_LINEAR = triangle with replicated borders. Output rounded to uint8 like cv2
(the fixed-point rounding of cv2 can differ by 1 LSB)."""
import functools

import numpy as np
import torch


def _area(n_in, n_out):
    s = n_in / n_out
    W = np.zeros((n_out, n_in))
    for o in range(n_out):
        a, b = o * s, (o + 1) * s
        i0, i1 = int(np.floor(a)), int(np.ceil(b))
        for i in range(i0, min(i1, n_in)):
            W[o, i] = max(0.0, min(b, i + 1) - max(a, i))
        W[o] /= W[o].sum()
    return W


def _kernel(n_in, n_out, kind):
    s = n_in / n_out
    W = np.zeros((n_out, n_in))
    for o in range(n_out):
        x = (o + 0.5) * s - 0.5
        x0 = int(np.floor(x))
        t = x - x0
        if kind == "cubic":
            A = -0.75
            ws = [((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
                  ((A + 2) * t - (A + 3)) * t * t + 1,
                  ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1]
            ws.append(1 - sum(ws))
            taps = range(x0 - 1, x0 + 3)
        else:  # linear
            if x < 0:
                x0, t = 0, 0.0
            ws, taps = [1 - t, t], range(x0, x0 + 2)
        for i, w in zip(taps, ws):
            W[o, min(max(i, 0), n_in - 1)] += w
    return W


@functools.lru_cache(maxsize=64)
def _mat(n_in, n_out, kind, device):
    if n_in == n_out:
        return None
    if kind == "auto":
        kind = "area" if n_out < n_in else "cubic"
    W = _area(n_in, n_out) if kind == "area" else _kernel(n_in, n_out, kind)
    return torch.tensor(W, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=64)
def _taps(n_in, n_out, kind, device):
    """sparse form of _mat: (idx [n_out, T] int64, wts [n_out, T] fp32), T = max nonzeros per output (4 for cubic)."""
    M = _mat(n_in, n_out, kind, "cpu")
    if M is None:
        return None
    nz = M != 0
    T = int(nz.sum(1).max())
    idx = torch.zeros(n_out, T, dtype=torch.long)
    wts = torch.zeros(n_out, T)
    for o in range(n_out):
        j = torch.nonzero(nz[o]).flatten()
        idx[o, :len(j)] = j
        wts[o, :len(j)] = M[o, j]
    return idx.to(device), wts.to(device)


def cv2_resize(img, h, w, kind="auto", round_u8=True):
    """img: [C, H, W] or [B, C, H, W] uint8/float tensor on GPU -> resized float32 (uint8-rounded values if round_u8).
    kind: 'auto' (= cv2 INTER_AREA when shrinking that axis, INTER_CUBIC when enlarging, as upstream DA3), 'area',
    'cubic', 'linear'. Both axes use the same rule as cv2 (cv2 picks AREA/CUBIC for the whole image: pass kind explicitly
    when the upstream code decides per image)."""
    x = img.float()
    H, W = x.shape[-2:]
    # (trtgap) separable resize as T-tap gathers (same weights as the dense matrices; the dense fp32 GEMMs ran on SIMT kernels:
    # 1.8 ms for a Midd-H image -> ~0.1 ms)
    tw, th = _taps(W, w, kind, x.device), _taps(H, h, kind, x.device)
    if tw is not None:
        idx, wts = tw
        x = (x[..., idx] * wts).sum(-1)
    if th is not None:
        idx, wts = th
        x = (x.transpose(-1, -2)[..., idx] * wts).sum(-1).transpose(-1, -2).contiguous()
    if round_u8:
        x = x.round_().clamp_(0, 255)
    return x


def resize_normalize_u8(img, h, w, kind, mean, std):
    """(trtgap) cv2_resize(img_u8, h, w, kind, round_u8=False) / 255, then (x - mean) / std, as fp16 [1, C, h, w], in one kernel
    (img: [C, H, W] uint8 CUDA, C <= 3). Same weights as cv2_resize; falls back to it for other inputs."""
    C, H, W = img.shape
    if img.dtype != torch.uint8 or C > 3:
        x = cv2_resize(img, h, w, kind, round_u8=False) / 255.0
        return ((x - mean.view(-1, 1, 1)) / std.view(-1, 1, 1))[None].half()
    from .kernels import Kernels
    ident = lambda n: (torch.arange(n, device=img.device).view(n, 1), torch.ones(n, 1, device=img.device))
    tx = _taps(W, w, kind, img.device) or ident(W)
    ty = _taps(H, h, kind, img.device) or ident(H)
    key = (W, w, H, h, kind)
    c = _I32.get(key)
    if c is None:
        c = _I32[key] = (tx[0].int().contiguous(), tx[1].float().contiguous(), ty[0].int().contiguous(), ty[1].float().contiguous())
    ixx, wxx, iyy, wyy = c
    out = torch.empty((1, C, h, w), device=img.device, dtype=torch.float16)
    m = [float(v) for v in mean.flatten().tolist()] + [0.0] * (3 - C)
    s = [float(v) for v in std.flatten().tolist()] + [1.0] * (3 - C)
    k = Kernels.get()
    k.launch("resize_norm_u8", k.linear(C * h * w), img, C, H, W, *img.stride(), ixx, wxx, ixx.shape[1], iyy, wyy, iyy.shape[1], h, w, *m, *s, out)
    return out


_I32 = {}
