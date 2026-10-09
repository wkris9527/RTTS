# Reproduction and provenance

## Source

The public release was prepared from [RTTS-70F4](https://anonymous.4open.science/r/RTTS-70F4/), whose source date is 2026-03-31. The original downloaded ZIP contained 170 files, including Python bytecode. The source-snapshot commit preserves 130 source/assets files, moves historical shell scripts to `scripts/legacy/`, and archives the initial README and dependency export. No datasets, model checkpoints, or local experiment logs are included.

The local RTTS directory's `ovss/clip/model.py` and `utils_local/logits_sam.py` were SHA-256-identical to the anonymous snapshot. A local VOC20 log recorded `method=mlmp`, `adapt=True`, ten optimizer steps, and 102,400 trainable parameters. It does not establish the provenance of every manuscript result.

## Two explicit execution settings

| Setting | Entry | Parameter updates | Feature aggregation |
| :--- | :--- | :--- | :--- |
| Training-free RTTS | `--method rtts`, no `--adapt` | None; every parameter is frozen | Final-layer NaCLIP features and averaged normalized text embeddings |
| Historical MLMP + region feedback | `--method mlmp --adapt --with_rtts` | Visual LayerNorm optimizer from archived MLMP | Historical multi-layer entropy weighting |

The new RTTS interface reuses the proposal/merging routines and extracts the snapshot's region voting and Sinkhorn feedback into a separate module. It handles Cityscapes patches one at a time, retains center-aware masked pooling, uses dynamic patch-grid dimensions, and exposes refinement rounds. It does not invent new model checkpoints or optimize on test labels.

The original snapshot enabled region feedback inside its CLIP constructor, so several scripts named as baselines inherited that configuration; TPT had its own region-feedback path. This release disables feedback by default for baselines. Explicit archived combinations are supported for MLMP and TPT only; the other methods require separate integration. Use the source-snapshot commit if exact historical execution is needed; `scripts/legacy/` alone does not reproduce the historical behavior of a changed backbone.

## Numerical details retained from code

Defaults include 10 semantic prompt points, response ratio 0.5, minimum response probability 0.1, minimum mask area 100, enhanced proposal selection, semantic quality weight 0.3, geometric weight 0.7, NMS IoU 0.5, and three Sinkhorn normalization steps. Dense feedback uses the archived `logits + 0.5 * (voted_logits + region_logits)` operation.

The manuscript describes normalized region prototypes, while the archived assignment path uses center-aware pooled patch features without an additional prototype normalization. This release preserves the archived operation. Likewise, the manuscript's generic high-resolution description omits the archived Cityscapes resize of 1120 x 560. These differences and the gradient-based historical configuration must be reconciled with the final experiment protocol before asserting exact Table I reproduction.

## What has and has not been verified

Lightweight checks validate parsing, preset generation, syntax, metric aggregation, and the core tensor operations. Tensor tests compare extracted Sinkhorn normalization and region feedback to the original operations. They use synthetic proposals and do not replace SAM or dataset evaluation.

The main evaluation environment and weights were unavailable during release preparation. Consequently, no full-dataset result, speed figure, or numerical equality between the dedicated RTTS interface and the manuscript table is asserted. Tables in README are manuscript-reported references. Optional SAM2/SAM3/CorrCLIP experiments and the additional manuscript ablations are not covered by the default reproduction interface.

## Recording an experiment

1. Record the Git commit and run `scripts/check_environment.py --output outputs/<run>/environment.json`.
2. Verify dataset annotation IDs and validation split.
3. Use separate output directories and retain commands, checkpoint hashes, seeds, and per-corruption scores.
4. Complete full validation rather than `--debug` or `--profile_max_images`.
5. Report clean mIoU separately from the unweighted mean over all 15 corruption types.
6. For timings, synchronize CUDA and report whether preprocessing, SAM encoding, and all patches are included. The archived runtime collection is not a publication-ready benchmark by itself.
