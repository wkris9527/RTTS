# Installation

Use a dedicated environment. These instructions do not modify an existing research environment.

## Main evaluation stack

The versions below retain the anonymous project's reported Python/PyTorch/MMCV/MMSegmentation stack and add missing direct dependencies. Full end-to-end compatibility remains to be benchmark-validated; see [validation](validation.md).

```bash
conda create -n rtts python=3.10.13 -y
conda activate rtts
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -r requirements.txt
python -m pip install -r requirements/sam.txt
python scripts/download_checkpoints.py
python scripts/check_environment.py --output outputs/environment.json
```

MMCV is tied to the installed PyTorch and CUDA versions. Consult the [official MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) if a matching prebuilt wheel is unavailable. Install only one of `mmcv` and `mmcv-lite`; the main stack specifies `mmcv`.

SAM is pinned to upstream commit `dca509fe793f601edb92606367a655c15ac00fdf`. The model weights are separate from the Python package. The default path is `checkpoints/sam_vit_h_4b8939.pth`. CLIP weights download automatically to `~/.cache/clip` on first load; an internet connection is required unless already cached.

To use an existing SAM checkpoint:

```bash
python scripts/run.py --benchmark v20 --data-dir /path/to/VOC2012 \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --output outputs/v20
```

The downloader prints the actual SHA-256. Supply an independently obtained hash using `--sha256` to enforce it. No unverified checksum is claimed for Meta's checkpoint.

## Windows

The evaluation launcher is Python-based and does not require Bash. Run each command on one line in PowerShell and use `--workers 0`. Alternatively use `conda run -n rtts python ...` to select the environment explicitly. The model currently uses the first visible CUDA device; select physical GPUs with `--gpu` / `CUDA_VISIBLE_DEVICES`.

## Optional SAM2 experiments

`sam2/` and `utils_local/logits_sam2.py` preserve the anonymous experimental source. They are not selected by the main RTTS launcher. Follow the [official SAM2 environment requirements](https://github.com/facebookresearch/sam2) in a separate environment, then inspect `requirements/sam2-optional.txt`. The archived extension build and checkpoint paths need separate setup. This release does not supply a validated SAM2/SAM3 reproduction recipe or CorrCLIP integration.
