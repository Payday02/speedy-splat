# Speedy-Splat — ROCm/HIP Fork (RX 7900 XTX, WSL2)

This is a ROCm/HIP port of [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat), tested on an AMD Radeon RX 7900 XTX (gfx1100) under WSL2 Ubuntu 24.04 with ROCm 7.2. Training and rendering both work end-to-end; no functional issues found beyond the fixes documented below.

Original project: [Alex Hanson](https://www.cs.umd.edu/~hanson/), [Allen Tu](https://tuallen.github.io), [Geng Lin](https://www.cs.umd.edu/people/geng), [Vasu Singla](https://vasusingla.github.io/), [Matthias Zwicker](https://www.cs.umd.edu/~zwicker/), [Tom Goldstein](https://www.cs.umd.edu/~tomg/) — [arXiv](https://arxiv.org/abs/2412.00578) | [Website](https://speedysplat.github.io)

## What's different in this fork

Both CUDA submodules were ported to HIP so they build and run under ROCm:

- **[diff-gaussian-rasterization](https://github.com/Payday02/speedy-splat-rasterizer/tree/rocm-port)** (`rocm-port` branch) — speedy-splat's modified rasterizer, patched to compile under hipcc:
  - Removed `device_launch_parameters.h` includes (CUDA-toolkit-only header, not needed under HIP)
  - Replaced `__trap()` with `__builtin_trap()` (HIP has no `__trap` intrinsic)
  - Removed unused `cooperative_groups/reduce.h` includes (never actually called, and unavailable on this ROCm version)
  - Normalized spaced `<< <...>> >` kernel launch syntax to `<<<...>>>` (hipcc's parser is stricter about this than nvcc's)
- **[simple-knn](https://github.com/Payday02/simple-knn-rocm/tree/rocm-port)** (`rocm-port` branch, mirrored from the original [Inria GitLab](https://gitlab.inria.fr/bkerbl/simple-knn)) — same include/chevron fixes, plus:
  - Added missing `#include <cfloat>` for `FLT_MAX`
  - Added `simple_knn/__init__.py` — required for pip's editable-install finder to resolve the package (this is a real correctness fix, not ROCm-specific; the upstream repo is missing this file too)

Both submodules build with PyTorch's standard automatic hipification (`pip install -e . --no-build-isolation`) once these patches are applied — no manual `hipify-clang` invocation needed.

## ROCm/WSL2 environment setup

Brief version — see AMD's [WSL install docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/) for the full walkthrough:

1. WSL2 Ubuntu 24.04, with a current Adrenalin driver (26.2.2+) on the Windows host for WSL compute support
2. `amdgpu-install` with `--usecase=wsl,rocm --no-dkms` — note the WSL usecase requires the **unversioned** package path (e.g. `.../amdgpu-install/7.2/...`, not `.../7.2.4/...`) or `--list-usecase` won't show `wsl` as an option
3. PyTorch: the stable ROCm wheel index only tracks ROCm 6.3/6.4; if your system ROCm is 7.x, install a nightly build instead, matched to your Python version (3.12, not 3.13 — ROCm wheels aren't validated for 3.13):
   ```bash
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
   ```
4. **Critical WSL-specific fix**: PyTorch's ROCm wheel bundles its own `libhsa-runtime64.so`, built for native Linux — it can't talk to WSL's `/dev/dxg` GPU shim, so `torch.cuda.is_available()` returns `False` even though the GPU is otherwise working. Swap in the WSL-aware system library:
   ```bash
   cd $(pip show torch | grep Location | awk -F ": " '{print $2}')/torch/lib/
   cp /opt/rocm/lib/libhsa-runtime64.so.<version-on-your-system> libhsa-runtime64.so
   ```
   (Find the exact filename with `ls -la /opt/rocm/lib/libhsa-runtime64.so*` — the version suffix varies by ROCm release.) This has to be redone after any fresh `pip install torch`.
5. **MIOpen convolution workaround** — SSIM loss computation (`F.conv2d`) can hit `MIOpen(HIP): Error [EvaluateInvokers] ... Invalid elapsed time detected`, a known upstream bug on gfx110x cards ([ROCm/TheRock#4681](https://github.com/ROCm/TheRock/issues/4681)) in MIOpen's convolution-algorithm benchmarking. Fix by skipping the exhaustive timing-based search:
   ```bash
   export MIOPEN_FIND_MODE=FAST
   export MIOPEN_FIND_ENFORCE=NONE
   ```
   Set these before every training/eval run (e.g. in your shell profile) until upstream fixes the underlying MIOpen issue.

### Known cosmetic issue

Speedy-splat's own kernel-timing instrumentation (`kernel_times`, and the "Training Time"/FPS lines printed during training) reports negative values under this ROCm build — almost certainly a CUDA-`Event`-based timer that doesn't translate cleanly through HIP on this driver stack. This doesn't affect training correctness (loss, PSNR, and SSIM all improve normally over training) — only the self-reported speed metrics are wrong. Not yet root-caused.

## Setup (general)

Follow the setup instructions for the original [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) codebase for anything not ROCm-specific. Speedy-Splat's code changes are in (1) the differential renderer submodule and (2) the Python files in this repo; this fork's changes are additionally in the HIP portability patches described above.

## Running

Set `SCENE_DATA_PATH` and `SCENE_MODEL_PATH`, then:

```shell
bash train.sh
```

- `SCENE_DATA_PATH` — path to the COLMAP or NeRF Synthetic dataset
- `SCENE_MODEL_PATH` — path where the model will be saved

Follows the 3D-GS pipeline; Tensorboard logging includes all metrics reported in the Speedy-Splat paper.

Note: this trains at a higher image resolution than the original 3D-GS `full_eval.py` protocol, resulting in larger models and a more demanding evaluation setting.

### Scene Metrics

To compute metrics for an already-trained model, set `SCENE_DATA_PATH`, `SCENE_MODEL_PATH`, and `ONLY_RAW_KERNEL_TIMES`, then:

```shell
bash compute_scene_metrics.sh
```

- `SCENE_DATA_PATH` — path to the COLMAP or NeRF Synthetic dataset
- `SCENE_MODEL_PATH` — path to the pretrained model
- `ONLY_RAW_KERNEL_TIMES` — `true`/`false`; if `true`, only raw per-image kernel time is returned

Metrics are written to `SCENE_MODEL_PATH/<train|test>/ours_<iteration>/metrics.csv`. If `ONLY_RAW_KERNEL_TIMES=true`, a `kernel_times.csv` is written instead, recording raw render time in milliseconds per image, repeated 20 times.

## Results

<img src="assets/speedy_splat_comparison.png" alt="Speedy Splat Comparison"/>

*3D Gaussian Splatting (3D-GS) is a recent 3D scene reconstruction technique that enables real-time rendering of novel views by modeling scenes as parametric point clouds of differentiable 3D Gaussians. However, its rendering speed and model size still present bottlenecks, especially in resource-constrained settings. Speedy-Splat optimizes the rendering pipeline to precisely localize Gaussians and introduces a novel pruning technique integrated into training, accelerating average rendering speed by 6.71x across Mip-NeRF 360, Tanks & Temples, and Deep Blending scenes with 10.6x fewer primitives than 3D-GS.*

## BibTeX

```
@InProceedings{HansonSpeedy,
    author    = {Hanson, Alex and Tu, Allen and Lin, Geng and Singla, Vasu and Zwicker, Matthias and Goldstein, Tom},
    title     = {Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21537-21546},
    url       = {https://speedysplat.github.io/}
}
```

## Related Work

- [PUP-3DGS](https://pup3dgs.github.io/) (CVPR 2025) — prunes 90% of primitives from any pretrained 3D-GS model using a mathematically principled sensitivity score, more than tripling rendering speed while retaining higher visual fidelity at a substantially higher compression ratio.
- [SpeeDe3DGS](https://speede3dgs.github.io/) — boosts DeformableGS rendering speed from 20 to 276 FPS using temporal sensitivity pruning and groupwise SE(3) motion distillation, preserving the image quality of per-Gaussian neural motion.

## Funding and Acknowledgments

This work was made possible by the IARPA WRIVA Program, the ONR MURI program, and DARPA TIAMAT. Commercial support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Further support was provided by the National Science Foundation (IIS-2212182), and by the NSF TRAILS Institute (2229885). Zwicker was additionally supported by the National Science Foundation (IIS-2126407).