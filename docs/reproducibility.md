# Reproduction protocol and source provenance

## Source and release scope

The sole source for algorithm code is [RTTS-70F4](https://anonymous.4open.science/r/RTTS-70F4/), downloaded from its public ZIP. Local research variants are not included. Python bytecode and temporary maintenance files are omitted; historical shell scripts are moved from `bash/` to `scripts/legacy/`. All 130 retained original files are listed with SHA-256 in [source-manifest.json](source-manifest.json), normalizing text line endings only. Model, adaptation, metrics, transforms, labels, seed handling, and evaluation code are unchanged.

## Exact released execution setting

The anonymous shell files named `rtts.sh` select `--method mlmp --adapt`. The original method registry has no `rtts` key, although the anonymous README uses that name in its example. The portable launcher follows the executable shell files and supplies the supported method key.

Defaults are NaCLIP ViT-L/14, vision outputs -1 through -18, alpha_cls 1.0, batch size 1, learning rate 0.001, ten optimization steps, one trial, seed 0, class extensions, and loss logging. Region feedback is enabled in the original CLIP constructor with two refinement rounds. SAM loads `sam_vit_h_4b8939.pth` from the working directory. No replacement training-free evaluator or new refinement implementation is included.

The manuscript describes a training-free approach, whereas the released shell configuration requests gradient adaptation. This release faithfully preserves the anonymous code and documents that difference; it does not reinterpret the experiment or claim those two settings are equivalent. Original baseline scripts also inherit the backbone's region-feedback flag.

## Data and metrics

Only validation splits are used. The standard resize is 224 x 224; Cityscapes uses the original 1120 x 560 resize and 224 x 224 patches with stride 112. Label remapping and evaluation resolution remain as implemented in the original dataset/transforms code. Corruptions are generated at severity 5. Report clean mIoU separately from the unweighted mean of all 15 corruptions; the summary utility omits that average when any corruption is missing.

## Recording a reproduction run

1. Follow [installation](installation.md) and [dataset preparation](datasets.md).
2. Save `scripts/check_environment.py --output outputs/environment.json` and the downloader's checkpoint SHA-256.
3. Run `scripts/run.py` with the desired preset and an unused output directory. Preserve `launch.json`, configurations, and per-corruption results.
4. Use the complete validation split and retain the configured trial count and seed.
5. Export the results using `scripts/summarize_results.py`.

The original evaluator's generated `cmd.sh` references `main_segmentation.py`; use the argument list recorded in `launch.json` or rerun the launcher instead. This known source issue is documented without editing the original file.

Full SAM inference and dataset scores have not been rerun for this release. README values are transcribed from the manuscript. Setup, syntax, source consistency, and reporting checks are described in [validation](validation.md). Optional SAM2 experiments require their own setup.
