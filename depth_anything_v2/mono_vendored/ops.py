# vendored from experiments/004-mono-depth: mono_fast/ops.py @ 2171802 (master c803da8) -- do not edit here
"""Small fused ops on fp16 NHWC (channels-last) tensors, size-independent prebuilt kernels (mono_fast/kernels.py)."""
import torch
import torch.nn.functional as F

from .kernels import Kernels


def interp_bilinear(x, size, align_corners):
    """F.interpolate(x, size, 'bilinear', align_corners) for fp16 channels-last x with C % 8 == 0 via one vectorized kernel
    (torch's NHWC bilinear kernel is several x slower); same arithmetic as PyTorch's (fp32 weights, one rounding)."""
    N, C, Hs, Ws = x.shape
    Hd, Wd = int(size[0]), int(size[1])
    if x.dtype != torch.float16 or C % 8 or not x.is_contiguous(memory_format=torch.channels_last):
        return F.interpolate(x, size=(Hd, Wd), mode="bilinear", align_corners=align_corners)
    if align_corners:
        sh = (Hs - 1) / (Hd - 1) if Hd > 1 else 0.0
        sw = (Ws - 1) / (Wd - 1) if Wd > 1 else 0.0
    else:
        sh, sw = Hs / Hd, Ws / Wd
    y = torch.empty((N, C, Hd, Wd), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    k = Kernels.get()
    k.launch("bilerp_nhwc", k.linear(N * Hd * Wd * C // 8), x, N, Hs, Ws, y, Hd, Wd, C, float(sh), float(sw), 1 if align_corners else 0)
    return y


def pad_replicate(x, p, relu=False):
    """F.pad(relu(x) if relu else x, (p, p, p, p), mode="replicate") for fp16 channels-last x (C % 8 == 0) in one kernel;
    returns a channels-last tensor."""
    N, C, H, W = x.shape
    if x.dtype != torch.float16 or C % 8 or not x.is_contiguous(memory_format=torch.channels_last):
        y = F.relu(x) if relu else x
        return F.pad(y, (p, p, p, p), mode="replicate")
    y = torch.empty((N, C, H + 2 * p, W + 2 * p), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    k = Kernels.get()
    k.launch("pad_rep_nhwc", k.linear(N * (H + 2 * p) * (W + 2 * p) * C // 8), x, N, H, W, C, p, 1 if relu else 0, y)
    return y

