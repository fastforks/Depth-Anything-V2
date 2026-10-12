"""Depth Anything V2 (S/B/L) with custom kernels: same math as `DepthAnythingV2.infer_image`, any input size, nothing compiled
per size.

    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.fast_kernels import FastDAv2
    fast = FastDAv2(model.cuda().eval())                 # converts the model to fp16 in place
    depth = fast.infer(img_rgb_u8_chw_cuda, 518)         # [3, H, W] uint8 CUDA -> relative disparity [H, W] fp32 CUDA

- GPU in / GPU out: the cv2 INTER_CUBIC resize (lower bound `input_size`, multiple of 14, on the float image) and the
  normalization run in one CUDA kernel
- DINOv2 in pure fp16: attention through the vendored fp16-accumulate flash kernel (`attn="hh"`; None = torch SDPA),
  residual + LayerScale as addcmul, optional fp16-accumulate cuBLAS GEMMs (`gacc`)
- DPT head fp16 channels-last: 1x1 out_conv before the bilinear upsample (exact reorder), NHWC bilinear kernel, optional
  fp16-accumulate cuDNN convolutions (`hacc`, needs `pip install nvidia-cudnn-frontend`); last 1x1 conv + ReLU in fp32
`kernels=False` (or a failure to load the prebuilt kernels, e.g. on a non-NVIDIA device) selects plain PyTorch for all of it.
"""
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mono_vendored.resize import cv2_resize, resize_normalize_u8
from .mono_vendored.vit import fp16_accumulate, sdpa, with_attn
from .mono_vendored import ops

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _attn_forward(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    o = sdpa(q, k, v)  # scale = head_dim ** -0.5 = upstream self.scale
    return self.proj(o.transpose(1, 2).reshape(B, N, C))


def _block_forward(self, x):
    a = self.attn(self.norm1(x))
    x = torch.addcmul(x, a, self.ls1.gamma) if hasattr(self.ls1, "gamma") else x + a
    m = self.mlp(self.norm2(x))
    return x.addcmul_(m, self.ls2.gamma) if hasattr(self.ls2, "gamma") else x.add_(m)


class _F32Conv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv.float()

    def forward(self, x):
        return self.conv(x.float())


class FastDAv2:
    def __init__(self, model, gacc=True, hacc=False, kernels=True, attn="hh"):
        """gacc: fp16-accumulate GEMMs in the ViT; hacc: fp16-accumulate cuDNN convs in the head (cudnn-frontend);
        kernels: use the prebuilt CUDA kernels (resize/normalize, NHWC bilinear, flash attention `attn`)."""
        from .dinov2_layers.attention import Attention
        from .dinov2_layers.block import Block
        from .util.blocks import FeatureFusionBlock, ResidualConvUnit
        if kernels:
            try:
                from .mono_vendored.kernels import Kernels
                Kernels.get()
            except Exception as e:  # no prebuilt kernel for this GPU and no nvcc: plain PyTorch
                warnings.warn(f"FastDAv2: custom kernels unavailable ({e!r}); using plain PyTorch ops")
                kernels = False
        self.m = model.half().eval()
        self.gacc, self.kern = gacc, kernels
        for mod in model.modules():
            if isinstance(mod, Attention):
                mod.forward = types.MethodType(_attn_forward, mod)
            if isinstance(mod, Block):
                mod.forward = types.MethodType(_block_forward, mod)
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
            if isinstance(mod, FeatureFusionBlock):
                mod.forward = types.MethodType(self._fusion_forward(), mod)
            if isinstance(mod, ResidualConvUnit):
                assert not mod.bn
        # positional-embedding interpolation cached per token grid (weights are fixed at inference)
        vit = model.pretrained
        f_pe = vit.interpolate_pos_encoding
        pe_cache = {}

        def pe_cached(x, w, h):
            key = (x.shape[1], w, h, x.dtype)
            if key not in pe_cache:
                if len(pe_cache) >= 16:  # bounded: many distinct input sizes must not grow memory without limit
                    pe_cache.clear()
                pe_cache[key] = f_pe(x, w, h)
            return pe_cache[key]
        vit.interpolate_pos_encoding = pe_cached
        model.depth_head.scratch.output_conv2[2] = _F32Conv(model.depth_head.scratch.output_conv2[2])
        if hacc:
            from .fastbb_vendored.cudnn_conv import patch_convs
            patch_convs(model.depth_head)
        dev = next(model.parameters()).device
        self.mean, self.std = MEAN.to(dev), STD.to(dev)
        with_attn(self, "infer", attn if kernels else None)

    def _interp(self, x, size, align_corners):
        if self.kern:
            return ops.interp_bilinear(x, size, align_corners)
        return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=align_corners)

    def _fusion_forward(self):
        interp = self._interp

        def fwd(m, *xs, size=None):
            y = xs[0]
            if len(xs) == 2:
                y = y + m.resConfUnit1(xs[1])
            y = m.resConfUnit2(y)
            y = m.out_conv(y)  # 1x1 conv before the upsample (exact: both linear)
            if size is None and m.size is None:
                size = (y.shape[-2] * 2, y.shape[-1] * 2)
            elif size is None:
                size = m.size
            return interp(y, size, m.align_corners)
        return fwd

    @staticmethod
    def sizes(H, W, input_size):
        from .util.transform import Resize
        r = Resize(width=input_size, height=input_size, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
                   resize_method="lower_bound")
        w, h = r.get_size(W, H)
        return int(h), int(w)

    @torch.inference_mode()
    def infer(self, img_u8, input_size=518):
        """img_u8: [3, H, W] uint8 RGB CUDA -> relative disparity [H, W] fp32 CUDA (= infer_image output)."""
        H, W = img_u8.shape[-2:]
        h, w = self.sizes(H, W, input_size)
        if self.kern:   # upstream: cv2 cubic on the float image, normalize
            x = resize_normalize_u8(img_u8, h, w, "cubic", self.mean, self.std)
        else:
            x = cv2_resize(img_u8, h, w, "cubic", round_u8=False) / 255.0
            x = ((x - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1))[None].half()
        m = self.m
        ph, pw = h // 14, w // 14
        with fp16_accumulate(self.gacc):
            feats = m.pretrained.get_intermediate_layers(x, m.intermediate_layer_idx[m.encoder], return_class_token=True)
        hd = m.depth_head
        outs = []
        for i, f in enumerate(feats):
            t = f[0]
            if hd.use_clstoken:
                t = hd.readout_projects[i](torch.cat((t, f[1].unsqueeze(1).expand_as(t)), -1))
            t = t.view(t.shape[0], ph, pw, t.shape[-1]).permute(0, 3, 1, 2)  # channels-last view
            outs.append(hd.resize_layers[i](hd.projects[i](t)))
        s = hd.scratch
        l1, l2, l3, l4 = s.layer1_rn(outs[0]), s.layer2_rn(outs[1]), s.layer3_rn(outs[2]), s.layer4_rn(outs[3])
        p = s.refinenet4(l4, size=l3.shape[2:])
        p = s.refinenet3(p, l3, size=l2.shape[2:])
        p = s.refinenet2(p, l2, size=l1.shape[2:])
        p = s.refinenet1(p, l1)
        o = s.output_conv1(p)
        o = self._interp(o, (ph * 14, pw * 14), True)
        d = s.output_conv2(o)[:, 0]  # ends with ReLU (upstream's extra outer F.relu is idempotent)
        return F.interpolate(d[:, None].float(), (H, W), mode="bilinear", align_corners=True)[0, 0]
