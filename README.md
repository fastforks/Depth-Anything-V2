# Depth Anything V2: faster inference fork

> **You are on `cuda-kernels`.** Research fork of [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (base `a561b84`; `main` mirrors upstream). Two branches:
> - [`portable`](https://github.com/fastforks/Depth-Anything-V2/tree/portable): plain PyTorch, no build step, no custom kernels. Contains the changes that also go upstream as separate PRs (`pr/*`) plus `infer_tensor` (GPU in / GPU out) and an opt-in `prepare_half()`.
> - [`cuda-kernels`](https://github.com/fastforks/Depth-Anything-V2/tree/cuda-kernels): `portable` plus `FastDAv2` (`depth_anything_v2/fast_kernels.py`): prebuilt CUDA kernels (GPU resize + normalize, flash attention with fp16 accumulation, NHWC bilinear; SASS for sm_80/86/89/90/120 plus PTX, loaded through the driver API, no compiler needed) and optional fp16-accumulate cuDNN convolutions (`hacc`, needs `pip install nvidia-cudnn-frontend`). `kernels=False`, or a GPU without a loadable kernel, falls back to plain PyTorch ops.
>
> **Licences.** Code: Apache-2.0, as upstream ([LICENSE](LICENSE)). Weights are not included and not re-hosted; get them from the [upstream release](https://huggingface.co/depth-anything) and respect their licences: Depth-Anything-V2-**Small is Apache-2.0**; **Base, Large (and Giant, not yet released) are CC BY-NC 4.0 (non-commercial)** (upstream README, "LICENSE"). The numbers below use Small (ViT-S) and Large (ViT-L).

```python
import torch
from depth_anything_v2.dpt import DepthAnythingV2
model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load("depth_anything_v2_vitl.pth", map_location="cpu")); model = model.cuda().eval()
model.prepare_half()                                   # portable branch: fp16 weights, last head conv stays fp32
depth = model.infer_tensor(img_rgb_u8_chw_cuda, 518)    # [3, H, W] uint8 CUDA -> relative disparity [H, W] fp32 CUDA

# cuda-kernels branch: the custom-kernel path (converts the model to fp16 in place instead of prepare_half)
from depth_anything_v2.fast_kernels import FastDAv2
fast = FastDAv2(model)                                  # hacc=True: fp16-accumulate cuDNN convs (nvidia-cudnn-frontend)
depth = fast.infer(img_rgb_u8_chw_cuda, 518)
```
`infer_image` (numpy BGR in, numpy out) is unchanged and also works after `prepare_half()`; load checkpoints before `prepare_half()` (it renames the last head conv's state-dict keys). Tests: `DAV2_CHECKPOINT=/path/to/depth_anything_v2_vits.pth DAV2_ENCODER=vits python tests/test_fast.py` (and `tests/test_kernels.py`).

RTX 5060 Ti 16 GB, PyTorch 2.14.1+cu130, 2026-10-12; one 1482x994 image; cold process per row, median of 15 runs after warm-up; 518 = `input_size=518`, 1024 = long side 1024. Time includes the resize and the resize back, input and output on the GPU for the fork rows (upstream: `infer_image`-equivalent steps, output kept on the GPU). Peak memory = torch allocated / reserved. "startup / first call / new size" = process start to model ready / first call at 518 / first call at the second size (1024) in the same process (total latency of that call). NYU AbsRel = 654 NYUv2 test images, least-squares scale + shift alignment in disparity. `hacc` is slower than the default at ViT-S 518.

**ViT-L**

| version | ms 518 | ms 1024 | speed-up 518 ; 1024 (vs flags / vs upstream) | peak MB at 518 | peak MB at 1024 | startup / first call / new size | weights MB | NYU AbsRel |
|---|---|---|---|---|---|---|---|---|
| upstream (fp32) | 243.9 | 605.0 | 0.42x / 1.00x ; 0.44x / 1.00x | 2106 / 2528 | 3077 / 4916 | 5.5 s / 398 ms / 601 ms | 1279 | 0.04388 |
| upstream + flags (`cudnn.benchmark`, `model.half()` with half input) | 101.4 | 267.5 | 1.00x / 2.41x ; 1.00x / 2.26x | 1080 / 1268 | 1568 / 2106 | 5.4 s / 1102 ms / 988 ms | 640 | 0.04388 |
| **portable**, fp32 | 204.8 | 483.7 | 0.50x / 1.19x ; 0.55x / 1.25x | 2112 / 2492 | 2740 / 3190 | 5.4 s / 396 ms / 486 ms | 1279 | 0.04388 |
| **portable** + `prepare_half()` | 55.9 | 113.9 | 1.81x / 4.36x ; 2.35x / 5.31x | 1081 / 1742 | 1394 / 1842 | 5.5 s / 411 ms / 118 ms | 640 | 0.04388 |
| **cuda-kernels** | 38.6 | 75.0 | 2.63x / 6.32x ; 3.57x / 8.06x | 992 / 1546 | 1240 / 1458 | 5.4 s / 397 ms / 106 ms | 640 | 0.04388 |
| **cuda-kernels** + `hacc` | 34.9 | 69.4 | 2.91x / 7.00x ; 3.85x / 8.72x | 993 / 1546 | 1247 / 1458 | 5.5 s / 338 ms / 134 ms | 640 | 0.04388 |

**ViT-S**

| version | ms 518 | ms 1024 | speed-up 518 ; 1024 (vs flags / vs upstream) | peak MB at 518 | peak MB at 1024 | startup / first call / new size | weights MB | NYU AbsRel |
|---|---|---|---|---|---|---|---|---|
| upstream (fp32) | 49.1 | 106.1 | 0.66x / 1.00x ; 0.62x / 1.00x | 403 / 556 | 795 / 1110 | 2.3 s / 354 ms / 110 ms | 95 | 0.05094 |
| upstream + flags (`cudnn.benchmark`, `model.half()` with half input) | 32.2 | 66.2 | 1.00x / 1.52x ; 1.00x / 1.60x | 225 / 372 | 425 / 580 | 2.2 s / 993 ms / 594 ms | 47 | 0.05093 |
| **portable**, fp32 | 27.0 | 61.7 | 1.19x / 1.82x ; 1.07x / 1.72x | 409 / 544 | 628 / 850 | 2.2 s / 371 ms / 62 ms | 95 | 0.05094 |
| **portable** + `prepare_half()` | 8.7 | 16.7 | 3.70x / 5.64x ; 3.96x / 6.36x | 226 / 302 | 338 / 410 | 2.2 s / 417 ms / 28 ms | 47 | 0.05094 |
| **cuda-kernels** | 5.7 | 11.3 | 5.65x / 8.58x ; 5.86x / 9.37x | 224 / 260 | 335 / 392 | 2.2 s / 422 ms / 46 ms | 47 | 0.05093 |
| **cuda-kernels** + `hacc` | 6.5 | 10.8 | 4.95x / 7.59x ; 6.13x / 9.78x | 225 / 260 | 337 / 392 | 2.3 s / 372 ms / 99 ms | 47 | 0.05093 |

The fp32 rows keep upstream precision; the fp16 rows change it (NYU AbsRel within 1e-5 of upstream).

Flags rows: driver-side only, no source edit (`model.half()` with half input plus `cudnn.benchmark`; start-up / first call / new size include the autotune). channels_last was not measured (ViT backbone, only the patch-embedding conv would change). Raw: `results/2026-10-23_flags_older_forks/` of the publishing workspace.

Upstream here is the code path with `xformers` not installed (the `Attention` fallback). Not shipped: the CUTLASS fused-convolution extension of the original study (a torch-ABI-specific 29 MB binary; with it ViT-L takes 32.2 / 63.8 ms). Full results, commands and raw data: `results/2026-10-12_dav2_release/` of the publishing workspace.

---

<div align="center">
<h1>Depth Anything V2</h1>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup>
<br>
[**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

<sup>1</sup>HKU&emsp;&emsp;&emsp;<sup>2</sup>TikTok
<br>
&dagger;project lead&emsp;*corresponding author

<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>
</div>

This work presents Depth Anything V2. It significantly outperforms [V1](https://github.com/LiheYoung/Depth-Anything) in fine-grained details and robustness. Compared with SD-based models, it enjoys faster inference speed, fewer parameters, and higher depth accuracy.

![teaser](assets/teaser.png)


## News
- **2025-01-22:** [Video Depth Anything](https://videodepthanything.github.io) has been released. It generates consistent depth maps for super-long videos (e.g., over 5 minutes).
- **2024-12-22:** [Prompt Depth Anything](https://promptda.github.io/) has been released. It supports 4K resolution metric depth estimation when low-res LiDAR is used to prompt the DA models.
- **2024-07-06:** Depth Anything V2 is supported in [Transformers](https://github.com/huggingface/transformers/). See the [instructions](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for convenient usage.
- **2024-06-25:** Depth Anything is integrated into [Apple Core ML Models](https://developer.apple.com/machine-learning/models/). See the instructions ([V1](https://huggingface.co/apple/coreml-depth-anything-small), [V2](https://huggingface.co/apple/coreml-depth-anything-v2-small)) for usage.
- **2024-06-22:** We release [smaller metric depth models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) based on Depth-Anything-V2-Small and Base.
- **2024-06-20:** Our repository and project page are flagged by GitHub and removed from the public for 6 days. Sorry for the inconvenience.
- **2024-06-14:** Paper, project page, code, models, demo, and benchmark are all released.


## Pre-trained Models

We provide **four models** of varying scales for robust relative depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |


## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

If you do not want to clone this repository, you can also load our models through [Transformers](https://github.com/huggingface/transformers/). Below is a simple code snippet. Please refer to the [official page](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for more details.

- Note 1: Make sure you can connect to Hugging Face and have installed the latest Transformers.
- Note 2: Due to the [upsampling difference](https://github.com/huggingface/transformers/pull/31522#issuecomment-2184123463) between OpenCV (we used) and Pillow (HF used), predictions may differ slightly. So you are more recommended to use our models through the way introduced above.
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]
```

### Running script on *images*

```bash
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--grayscale]
```
Options:
- `--img-path`: You can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- `--input-size` (optional): By default, we use input size `518` for model inference. ***You can increase the size for even more fine-grained results.***
- `--pred-only` (optional): Only save the predicted depth map, without raw image.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### Running script on *videos*

```bash
python run_video.py \
  --encoder <vits | vitb | vitl | vitg> \
  --video-path assets/examples_video --outdir video_depth_vis \
  [--input-size <size>] [--pred-only] [--grayscale]
```

***Our larger model has better temporal consistency on videos.***

### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2).

***Note: Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this [issue](https://github.com/LiheYoung/Depth-Anything/issues/81)).*** In V1, we *unintentionally* used features from the last four layers of DINOv2 for decoding. In V2, we use [intermediate features](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169) instead. Although this modification did not improve details or accuracy, we decided to follow this common practice.


## Fine-tuned to Metric Depth Estimation

Please refer to [metric depth estimation](./metric_depth).


## DA-2K Evaluation Benchmark

Please refer to [DA-2K benchmark](./DA-2K.md).


## Community Support

**We sincerely appreciate all the community support for our Depth Anything series. Thank you a lot!**

- Apple Core ML:
    - https://developer.apple.com/machine-learning/models
    - https://huggingface.co/apple/coreml-depth-anything-v2-small
    - https://huggingface.co/apple/coreml-depth-anything-small
- Transformers:
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
- TensorRT:
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python
- ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- ComfyUI: https://github.com/kijai/ComfyUI-DepthAnythingV2
- Transformers.js (real-time depth in web): https://huggingface.co/spaces/Xenova/webgpu-realtime-depth-estimation
- Android:
  - https://github.com/shubham0204/Depth-Anything-Android
  - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything


## Acknowledgement

We are sincerely grateful to the awesome Hugging Face team ([@Pedro Cuenca](https://huggingface.co/pcuenq), [@Niels Rogge](https://huggingface.co/nielsr), [@Merve Noyan](https://huggingface.co/merve), [@Amy Roberts](https://huggingface.co/amyeroberts), et al.) for their huge efforts in supporting our models in Transformers and Apple Core ML.

We also thank the [DINOv2](https://github.com/facebookresearch/dinov2) team for contributing such impressive models to our community.


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
