<div align="center">

# RTTS

### Towards Robust Test-Time Segmentation via Iterative Object-centric Adaptation

**Junhui Yin · Wenzhe Wang · Bin Fan · Hongmin Liu**<br>
University of Science and Technology Beijing

[![Python](https://img.shields.io/badge/Python-3.10-blue)](docs/installation.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Checks](https://github.com/wkris9527/RTTS/actions/workflows/checks.yml/badge.svg)](https://github.com/wkris9527/RTTS/actions/workflows/checks.yml)

**Object-centric refinement · Open-vocabulary semantic segmentation · Corruption robustness**

</div>

RTTS improves dense vision-language predictions under domain shift through iterative object-level refinement. CLIP semantic responses guide SAM proposals; spatial and semantic cues consolidate regions; Sinkhorn-based assignment provides semantic feedback for the next round.

![RTTS framework](figures/overview.jpg)

Code release from the [original anonymous repository](https://anonymous.4open.science/r/RTTS-70F4/) for **Towards Robust Test-Time Segmentation via Iterative Object-centric Adaptation**, with evaluation presets for seven segmentation benchmarks.

See [installation](docs/installation.md), [dataset preparation](docs/datasets.md), [reproduction protocol](docs/reproducibility.md), and [validation scope](docs/validation.md).

## Getting started

### 1. Install the environment

The main evaluation stack uses Python 3.10, PyTorch 2.1.2 / CUDA 11.8, NaCLIP ViT-L/14, and SAM ViT-H. Linux with an NVIDIA GPU is the reference installation target. The launcher also supports Windows; use `--workers 0` if multiprocessing is unavailable.

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

### 2. Download the model weights

```bash
python scripts/download_checkpoints.py
python scripts/check_environment.py --output outputs/environment.json
```

Place the SAM ViT-H checkpoint at `sam_vit_h_4b8939.pth` in the repository root, as expected by the anonymous implementation. The downloader uses this path. CLIP ViT-L/14 weights download automatically on first use.

### 3. Prepare the validation data

Follow [dataset preparation](docs/datasets.md). For the first evaluation, the VOC root must contain:

```text
VOC2012/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/val.txt
```

Pass the dataset root directly to `--data-dir`. Corruptions are generated during evaluation; no separate corrupted dataset download is required.

### 4. Run evaluation

```bash
# PASCAL VOC20, clean validation images
python scripts/run.py --benchmark v20 --data-dir /path/to/VOC2012 \
  --output outputs/v20-clean --gpu 0

# Clean images and 15 corruptions at severity 5
python scripts/run.py --benchmark v20 --data-dir /path/to/VOC2012 \
  --output outputs/v20-all --corruptions all --gpu 0

# Preview the exact command without importing the ML stack
python scripts/run.py --benchmark cityscapes --data-dir /path/to/cityscapes \
  --output outputs/cityscapes --dry-run
```

The launcher preserves the anonymous `rtts.sh` setting: `--method mlmp --adapt`, NaCLIP ViT-L/14, 18 vision layers, batch size 1, learning rate 0.001, ten adaptation steps, and one trial. Region feedback remains enabled inside the original backbone with two rounds. The evaluator and algorithm files are unchanged; the launcher only supplies paths and experiment arguments. See [protocol notes](docs/reproducibility.md) for the distinction between the manuscript description and the released execution setting.

Use the same command with the following presets and dataset roots. Set `--corruptions all` for clean images plus the complete corruption suite, or omit it for clean evaluation. Use `--trials 3` to record three seeded trials.

| `--benchmark` | `--data-dir` example |
| :--- | :--- |
| `v20`, `v21` | `/path/to/VOC2012` |
| `p59`, `p60` | `/path/to/PASCALContext` |
| `cityscapes` | `/path/to/cityscapes` |
| `coco_obj`, `coco_stuff` | `/path/to/coco` |

### 5. Collect the results

The launcher records the actual argument list, GPU selection, preset, seed, and Git commit in `launch.json`. The original evaluator saves configurations and aggregate/per-corruption metrics. Seed handling and data ordering are kept exactly as in the anonymous implementation. Use a separate output directory for each experiment.

```text
outputs/v20-all/
├── launch.json
├── configurations.txt
├── cmd.sh
├── results.txt
├── 00_original/results.txt
└── ...
```

Export the clean score and the corruption average:

```bash
python scripts/summarize_results.py outputs/v20-all/results.txt \
  --output outputs/v20-all/summary.csv
```

## Benchmarks and reported results

Mean Intersection over Union (mIoU, %), reported in the manuscript's Table I. `O` denotes clean inputs; `C` is the mean over the 15 corruption types at severity 5. Background is included in VOC21 / Context60 and excluded in VOC20 / Context59. No corrupted COCO results are reported in this table.

| Benchmark | Classes | Clean (O) | Corrupted (C) |
| :--- | ---: | ---: | ---: |
| PASCAL VOC21 | 21 | 55.54 | 47.93 |
| PASCAL VOC20 | 20 | 85.30 | 76.84 |
| PASCAL Context59 | 59 | 35.31 | 28.63 |
| PASCAL Context60 | 60 | 30.94 | 25.52 |
| Cityscapes | 19 | 40.99 | 25.26 |
| COCO-Object | 80 | 31.96 | — |
| COCO-Stuff | 171 | 23.22 | — |

These are manuscript-reported results, not measurements from the release checks. Full benchmark scores have not been rerun for this release; the anonymous execution setting is documented in [reproduction notes](docs/reproducibility.md).

The archived dataset implementation defines 171 COCO-Stuff categories. “164k” refers to dataset image count, not its category count. Dataset class definitions are the authority for the released evaluator.

## Code map

| Location | Purpose |
| :--- | :--- |
| `adapt/mlmp.py` | Original evaluation and adaptation entry selected by `rtts.sh` |
| `ovss/clip/model.py` | Original region feedback, feature aggregation, and Sinkhorn assignment |
| `utils_local/logits_sam.py` | CLIP-guided proposals and spatial-semantic merging |
| `ovss/clip/` | CLIP and NaCLIP backbone implementation |
| `utils_local/segmentation_datasets.py` | Dataset labels, splits, and metadata |
| `utils_local/mm_transforms.py` | Corruptions, resize, patch extraction, normalization |
| `configs/` | Seven benchmark presets with required dataset paths |
| `scripts/` | Launch, download, diagnose, and summarize |
| `scripts/legacy/` | Archived anonymous experiment commands |
| `sam2/` | Optional archived SAM2 experiment code; excluded from the default path |
| `tests/` | Source consistency and launcher/reporting checks |

## Baselines and historical experiments

The algorithm files are taken exclusively from the anonymous repository. TENT, TPT, WATT, CLIPArTT, and MLMP remain as released there. The original backbone enables region feedback at construction, including for some scripts named as baselines; this behavior is preserved.

The original shell commands are archived under `scripts/legacy/`. The original README and dependency export are preserved for provenance. [source-manifest.json](docs/source-manifest.json) records SHA-256 for all 130 retained original files, normalizing text line endings only. Added setup and reporting tools do not change the model or evaluation code.

## Citation

If you use RTTS in your research, please cite the accompanying manuscript:

```bibtex
@unpublished{yin2026rtts,
  title  = {Towards Robust Test-Time Segmentation via Iterative Object-centric Adaptation},
  author = {Yin, Junhui and Wang, Wenzhe and Fan, Bin and Liu, Hongmin},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/wkris9527/RTTS}
}
```

Machine-readable software metadata is provided in [CITATION.cff](CITATION.cff).

## License and acknowledgments

Original RTTS contributions use the [MIT License](LICENSE). Incorporated third-party code retains its own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. We thank the authors of [CLIP](https://github.com/openai/CLIP), [NaCLIP](https://github.com/sinahmr/NACLIP), [SAM](https://github.com/facebookresearch/segment-anything), [MLMP](https://github.com/dosowiechi/MLMP), SAM2, TENT, WATT, TPT, MMSegmentation, and imagecorruptions.

For questions or reproducibility reports, use [GitHub Issues](https://github.com/wkris9527/RTTS/issues) with the command, environment report, and relevant log excerpt.

