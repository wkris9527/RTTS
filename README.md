<div align="center">

# RTTS

### Towards Robust Test-Time Segmentation via Iterative Object-centric Adaptation

**Junhui Yin\* · Wenzhe Wang\* · Bin Fan · Hongmin Liu**  
University of Science and Technology Beijing  
<sup>\* Equal contribution</sup>

[![Python](https://img.shields.io/badge/Python-3.10-blue)](docs/installation.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange)](docs/installation.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Overview](#overview) · [Installation](#installation) · [Datasets](#datasets) · [Evaluation](#evaluation) · [Results](#results) · [Citation](#citation)

</div>

Official PyTorch implementation of **RTTS**.

## Overview

RTTS addresses open-vocabulary semantic segmentation under test-time domain shift through iterative object-centric refinement. It combines the semantic cues of CLIP with the structural priors of SAM to improve spatial coherence and semantic consistency.

<p align="center">
  <img src="figures/overview.jpg" alt="Overview of RTTS" width="100%">
</p>

The framework consists of three components:

- **Category-aware region generation:** category-specific CLIP responses guide SAM to generate semantically relevant mask proposals.
- **Mask region consolidation:** spatial continuity and semantic consistency group fragmented proposals into coherent object-level regions.
- **Context-aware category assignment:** Sinkhorn–Knopp normalization stabilizes region-to-category assignment while retaining soft uncertainty. Refined semantics guide the next round of region generation.

## Installation

The evaluation stack uses **Python 3.10**, **PyTorch 2.1.2**, **CUDA 11.8**, and **MMSegmentation 1.2.2**. An NVIDIA GPU is required.

```bash
git clone https://github.com/wkris9527/RTTS.git
cd RTTS

conda create -n rtts python=3.10.13 -y
conda activate rtts

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -r requirements.txt
python -m pip install -r requirements/sam.txt
```

Download the **SAM ViT-H** checkpoint and check the environment:

```bash
python scripts/download_checkpoints.py
python scripts/check_environment.py --output outputs/environment.json
```

The checkpoint should be placed at `RTTS/sam_vit_h_4b8939.pth`; the downloader uses this location automatically. **CLIP ViT-L/14** weights are downloaded on first use.

For platform-specific setup and troubleshooting, see [installation instructions](docs/installation.md). On Windows, use `--workers 0` in the evaluation commands.

## Datasets

Evaluation uses the **validation splits** of five datasets, covering seven benchmark settings. Prepare the images and segmentation labels using [dataset preparation instructions](docs/datasets.md), then pass the dataset root to `--data-dir`.

| Dataset | Benchmark preset | Classes |
| :--- | :--- | :--- |
| PASCAL VOC | `v20` / `v21` | 20 / 21 |
| PASCAL Context | `p59` / `p60` | 59 / 60 |
| Cityscapes | `cityscapes` | 19 |
| COCO-Object | `coco_obj` | 80 |
| COCO-Stuff | `coco_stuff` | 171 |

For example, the PASCAL VOC dataset root should have the following structure:

```text
VOC2012/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/
    └── Segmentation/
        └── val.txt
```

The VOC21 and Context60 settings include background; VOC20 and Context59 exclude it. The 15 corruption types are generated during evaluation at **severity 5**.

## Evaluation

Run all commands from the repository root. Replace the example dataset paths with your own.

### Clean evaluation

```bash
python scripts/run.py --benchmark v20 --data-dir /path/to/VOC2012 \
  --output outputs/v20-clean --gpu 0
```

### Corruption robustness

Evaluate clean images and all 15 corruption types:

```bash
python scripts/run.py --benchmark v20 --data-dir /path/to/VOC2012 \
  --output outputs/v20-all --corruptions all --gpu 0
```

To evaluate a single corruption, replace `all` with its name, for example `gaussian_noise`.

### Other benchmarks

Select a preset from the dataset table and supply the corresponding dataset root. For example:

```bash
python scripts/run.py --benchmark cityscapes --data-dir /path/to/cityscapes \
  --output outputs/cityscapes-all --corruptions all --gpu 0
```

Use `--trials 3` for three trials or `--dry-run` to inspect the command before evaluation. Each benchmark has a configuration under [configs/](configs/). The commands use **NaCLIP ViT-L/14** and **SAM ViT-H**; full execution settings are documented in the [evaluation protocol](docs/reproducibility.md).

### Collect results

Metrics are saved in the output directory, including aggregate `results.txt` and per-corruption results. The launcher also records the command and configuration in `launch.json`.

Export the results to CSV:

```bash
python scripts/summarize_results.py outputs/v20-all/results.txt \
  --output outputs/v20-all/summary.csv
```

The corruption score is the unweighted mean over all 15 corruption types, excluding clean images. The summary tool computes this score only when all 15 results are present.

## Results

**mIoU (%) reported in Table I of the manuscript.** Clean scores evaluate uncorrupted inputs; corruption scores average the 15 corruption types at severity 5.

| Benchmark | Clean | Corruption |
| :--- | ---: | ---: |
| PASCAL VOC21 | 55.54 | 47.93 |
| PASCAL VOC20 | 85.30 | 76.84 |
| PASCAL Context59 | 35.31 | 28.63 |
| PASCAL Context60 | 30.94 | 25.52 |
| Cityscapes | 40.99 | 25.26 |
| COCO-Object | 31.96 | — |
| COCO-Stuff | 23.22 | — |

A dash indicates that the corresponding result is not reported in Table I. See [evaluation protocol](docs/reproducibility.md) for experimental settings and reproduction notes.

## Citation

If you find this work useful, please cite:

```bibtex
@unpublished{yin2026rtts,
  title  = {Towards Robust Test-Time Segmentation via Iterative Object-centric Adaptation},
  author = {Yin, Junhui and Wang, Wenzhe and Fan, Bin and Liu, Hongmin},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/wkris9527/RTTS}
}
```

## Acknowledgments

This project builds on [MLMP](https://github.com/dosowiechi/MLMP), [NaCLIP](https://github.com/sinahmr/NaCLIP), [CLIP](https://github.com/openai/CLIP), and [SAM](https://github.com/facebookresearch/segment-anything). We thank their authors for making the code and models available.

## License

RTTS contributions are released under the [MIT License](LICENSE). Third-party components retain their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

For questions and reproducibility reports, please open a [GitHub issue](https://github.com/wkris9527/RTTS/issues).

